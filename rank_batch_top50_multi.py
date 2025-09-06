#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import orjson
import numpy as np
from tqdm import tqdm

# --------------------------- year buckets ---------------------------

YEAR_BUCKETS = [2019, 2020, 2021, 2022, 2023]

def pick_bucket_year(y: int) -> int:
    if y <= 2019: return 2019
    if y == 2020: return 2020
    if y == 2021: return 2021
    if y == 2022: return 2022
    return 2023

# --------------------------- utils ---------------------------

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x.astype(np.float32, copy=False)
    return (x / n).astype(np.float32, copy=False)

def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)

def sanitize_model_label(stem: str, base_prefix: str) -> str:
    """Из имени файла получить короткую метку модели. Пример: journal_softk@4,10 -> softk@4,10."""
    if base_prefix and stem.startswith(base_prefix + "_"):
        return stem[len(base_prefix) + 1 :]
    return stem

def should_skip(path: Path, exclude_substrings: List[str]) -> bool:
    name = path.name.lower()
    for s in exclude_substrings:
        s = s.strip().lower()
        if s and s in name:
            return True
    return False

def is_multi_vector_model(path: Path, probe_lines: int = 10) -> bool:
    """Берём только файлы, где есть поле 'embeddings' (а не 'embedding')."""
    try:
        with path.open("rb") as f:
            for _ in range(probe_lines):
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                if "embeddings" in rec:
                    return True
    except Exception:
        return False
    return False

# --------------------- годовые source_id (скан + кэш) ---------------------

def load_year_source_ids(year_file: Path) -> np.ndarray:
    """Из файла <year>.jsonl достаёт множество source_id, возвращает unique sorted np.int64."""
    sids = set()
    with year_file.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                continue
            sid = rec.get("source_id")
            if sid is not None:
                try:
                    sids.add(int(sid))
                except Exception:
                    continue
    return np.fromiter(sorted(sids), dtype=np.int64)

def ensure_src_cache(index_dir: Path) -> Path:
    p = index_dir / ".src_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_or_build_year_src_sets(index_dir: Path) -> Dict[int, np.ndarray]:
    """
    Для каждого Y из YEAR_BUCKETS ищет <index_dir>/<Y>.jsonl,
    строит (или грузит из кэша) массив source_id, присутствующих в этом файле.
    """
    cache_dir = ensure_src_cache(index_dir)
    out: Dict[int, np.ndarray] = {}
    any_found = False
    for y in YEAR_BUCKETS:
        jf = index_dir / f"{y}.jsonl"
        if not jf.exists():
            continue
        any_found = True
        npy = cache_dir / f"{y}_source_ids.npy"
        if npy.exists():
            try:
                out[y] = np.load(npy)
                continue
            except Exception:
                pass
        sids = load_year_source_ids(jf)
        np.save(npy, sids)
        out[y] = sids
    if not any_found:
        raise RuntimeError(f"В {index_dir} не найдено годовых файлов 2019..2023.")
    return out

# --------------------- загрузка мульти-модели (из jsonl) ---------------------

def load_multi_model_jsonl(path: Path,
                           expect_dim: int | None = None,
                           show_tqdm: bool = True):
    """
    Читает мульти-модель JSONL (v2).
    Возвращает: src_ids (S,), protos (P,D), groups (P,), priors (P,)
    Векторы L2-нормируются.
    """
    src_ids: List[int] = []
    protos: List[np.ndarray] = []
    groups: List[int] = []
    priors: List[float] = []

    total = count_lines(path) if show_tqdm else None
    it = tqdm(path.open("rb"), total=total, unit="lines", desc=f"Load model {path.name}") if show_tqdm else path.open("rb")
    with it as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = orjson.loads(raw)
            if "embeddings" not in rec:
                continue
            src = int(rec["source_id"])
            embs = rec.get("embeddings") or []
            if not embs:
                continue
            w = rec.get("weights", None)
            if w is not None and len(w) != len(embs):
                w = None

            src_idx = len(src_ids)
            src_ids.append(src)
            for i, e in enumerate(embs):
                v = np.asarray(e, dtype=np.float32)
                if expect_dim is not None and v.size != expect_dim:
                    continue
                protos.append(l2norm(v))
                groups.append(src_idx)
                priors.append(float(w[i]) if w is not None else 1.0)

    if not protos:
        raise RuntimeError(f"Модель пуста или несовместима по размерности: {path}")

    protos_mat = np.vstack(protos).astype(np.float32, copy=False)
    group_idx  = np.asarray(groups, dtype=np.int32)
    priors_arr = np.asarray(priors, dtype=np.float32)
    src_arr    = np.asarray(src_ids, dtype=np.int32)
    return src_arr, protos_mat, group_idx, priors_arr

# --------------------- упаковка модели в pack ---------------------

def pack_multi_model(jsonl_path: Path,
                     pack_dir: Path,
                     year_src_sets: Dict[int, np.ndarray],
                     expect_dim: int) -> Path:
    """
    Строит pack (.npz без компрессии) для мульти-модели:
      - protos_sorted [P,D] по группам
      - groups_sorted [P] (0..S-1)
      - priors_sorted [P]
      - src_ids [S]
      - ptr_full [S+1] — CSR-пойнтеры начала/конца прототипов каждого source
      - для каждого года y:
          perm_y [Py] — индексы столбцов protos_sorted, разрешённых для года
          gidx_y [Gy] — список индексов источников (0..S-1) в том же порядке
          ptr_y  [Gy+1] — CSR-пойнтеры по perm_y
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    stem = jsonl_path.stem
    pack_path = pack_dir / f"{stem}.npz"
    if pack_path.exists():
        return pack_path

    src_ids, protos, groups, priors = load_multi_model_jsonl(jsonl_path, expect_dim=expect_dim, show_tqdm=True)
    S = int(src_ids.size)

    # сортировка по группам, построение ptr_full
    order = np.argsort(groups, kind="stable")
    protos_sorted = protos[order]
    groups_sorted = groups[order]
    priors_sorted = priors[order]

    counts = np.bincount(groups_sorted, minlength=S).astype(np.int64)
    ptr_full = np.zeros(S + 1, dtype=np.int64)
    ptr_full[1:] = np.cumsum(counts)

    # map source_id -> group index (0..S-1)
    sid2g = {int(sid): i for i, sid in enumerate(src_ids.tolist())}

    # пер-годовые структуры
    year_arrays = {}
    for y, allowed_sids in year_src_sets.items():
        gidx = [sid2g[s] for s in allowed_sids.tolist() if s in sid2g]
        if not gidx:
            perm_y = np.array([], dtype=np.int64)
            gidx_y = np.array([], dtype=np.int32)
            ptr_y  = np.array([0], dtype=np.int64)
        else:
            gidx_y = np.asarray(sorted(set(gidx)), dtype=np.int32)
            parts = [np.arange(ptr_full[g], ptr_full[g+1], dtype=np.int64) for g in gidx_y.tolist()]
            perm_y = np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.int64)
            cnts = counts[gidx_y]
            ptr_y = np.zeros(gidx_y.size + 1, dtype=np.int64)
            ptr_y[1:] = np.cumsum(cnts)

        year_arrays[y] = (perm_y, gidx_y, ptr_y)

    # сохраняем без компрессии (быстрая загрузка)
    np.savez(pack_path,
             protos=protos_sorted,
             groups=groups_sorted,
             priors=priors_sorted,
             src_ids=src_ids,
             ptr_full=ptr_full,
             **{f"perm_{y}": year_arrays[y][0] for y in YEAR_BUCKETS},
             **{f"gidx_{y}": year_arrays[y][1] for y in YEAR_BUCKETS},
             **{f"ptr_{y}":  year_arrays[y][2] for y in YEAR_BUCKETS})
    return pack_path

def load_pack(pack_path: Path):
    z = np.load(pack_path, allow_pickle=False)
    protos = z["protos"]; groups = z["groups"]; priors = z["priors"]
    src_ids = z["src_ids"]; ptr_full = z["ptr_full"]
    per_year = {}
    for y in YEAR_BUCKETS:
        keyp, keyg, keyr = f"perm_{y}", f"gidx_{y}", f"ptr_{y}"
        if keyp in z and keyg in z and keyr in z:
            per_year[y] = (z[keyp], z[keyg], z[keyr])
    return protos, groups, priors, src_ids, ptr_full, per_year

# --------------------- групповые агрегаторы (векторизовано) ---------------------

def pool_max(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """scores: [B, Py], ptr: [Gy+1] -> [B, Gy] по максимуму секций."""
    return np.maximum.reduceat(scores, ptr[:-1], axis=1)

def pool_mean(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    sums = np.add.reduceat(scores, ptr[:-1], axis=1)              # [B, Gy]
    cnts = (ptr[1:] - ptr[:-1]).astype(np.float32)[None, :]       # [1, Gy]
    return sums / np.maximum(cnts, 1.0)

def pool_softmax(scores: np.ndarray, ptr: np.ndarray, tau: float = 10.0,
                 priors_cols: np.ndarray | None = None) -> np.ndarray:
    """
    Векторизованный softmax-пуллинг по секциям (CSR).
    scores: [B, Py], ptr: [Gy+1]; возвращает [B, Gy].
    priors_cols: веса длины Py для столбцов (например, веса кластеров), опционально.
    """
    a = tau * scores                                              # [B, Py]
    # стабилизация по секциям
    m = np.maximum.reduceat(a, ptr[:-1], axis=1)                  # [B, Gy]
    seg = (ptr[1:] - ptr[:-1]).astype(np.int64)                   # [Gy]
    col_group = np.repeat(np.arange(seg.size, dtype=np.int64), seg)  # [Py]
    exps = np.exp(a - m[:, col_group])                            # [B, Py]
    if priors_cols is not None:
        exps *= priors_cols[None, :]
    den  = np.add.reduceat(exps,        ptr[:-1], axis=1)         # [B, Gy]
    num  = np.add.reduceat(exps*scores, ptr[:-1], axis=1)         # [B, Gy]
    return num / np.maximum(den, 1e-9)

def pool_topm(scores: np.ndarray, ptr: np.ndarray, m: int = 3) -> np.ndarray:
    """Честный top-m по секциям (медленнее; используется только если явно выбран)."""
    B = scores.shape[0]
    Gy = ptr.size - 1
    out = np.full((B, Gy), -np.inf, dtype=np.float32)
    for j in range(Gy):
        s, e = int(ptr[j]), int(ptr[j+1])
        if e <= s:
            continue
        S = scores[:, s:e]                 # [B, len]
        if S.shape[1] <= m:
            out[:, j] = S.mean(axis=1)
        else:
            k = m
            part = np.partition(S, kth=S.shape[1]-k, axis=1)[:, -k:]
            out[:, j] = part.mean(axis=1)
    return out

# --------------------- queries ---------------------

def query_iter_grouped_by_year(queries_path: Path,
                               total: int | None = None,
                               batch: int = 1024):
    """
    Стримит батчи запросов, сгруппированные по бакету года.
    Возвращает (year, ids[np.int64], Q[np.float32 (B,D)])
    """
    buckets: Dict[int, List[Tuple[int, np.ndarray]]] = {y: [] for y in YEAR_BUCKETS}
    with queries_path.open("rb") as f:
        it = tqdm(f, total=total, unit="q", desc="Load queries") if total else f
        for raw in it:
            if not raw.strip():
                continue
            rec = orjson.loads(raw)
            qid = rec.get("id"); emb = rec.get("embedding"); y = rec.get("date_year")
            if qid is None or emb is None or y is None:
                continue
            by = pick_bucket_year(int(y))
            buckets[by].append((int(qid), l2norm(np.asarray(emb, dtype=np.float32))))
            if sum(len(v) for v in buckets.values()) >= batch:
                for yr in YEAR_BUCKETS:
                    if not buckets[yr]:
                        continue
                    ids = np.asarray([t[0] for t in buckets[yr]], dtype=np.int64)
                    Q   = np.vstack([t[1] for t in buckets[yr]]).astype(np.float32, copy=False)
                    buckets[yr].clear()
                    yield yr, ids, Q
        # остатки
        for yr in YEAR_BUCKETS:
            if not buckets[yr]:
                continue
            ids = np.asarray([t[0] for t in buckets[yr]], dtype=np.int64)
            Q   = np.vstack([t[1] for t in buckets[yr]]).astype(np.float32, copy=False)
            buckets[yr].clear()
            yield yr, ids, Q

# --------------------- поиск моделей ---------------------

def find_models(models_dir: Path,
                base_prefix: str | None,
                exclude_substrings: List[str]) -> List[Path]:
    if not models_dir.exists():
        return []
    all_files = sorted(models_dir.glob("*.jsonl"))
    prefixed = [p for p in all_files if base_prefix and p.stem.startswith(base_prefix + "_")]
    others   = [p for p in all_files if not (base_prefix and p.stem.startswith(base_prefix + "_"))]
    files: List[Path] = []
    for p in prefixed + others:
        if should_skip(p, exclude_substrings):
            continue
        if is_multi_vector_model(p):
            files.append(p)
    return files

# --------------------------- CLI ---------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Batch Top-K для мультивекторных моделей с фильтром по годам и упаковкой pack."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--queries", type=str,
                   help="Путь к *_test_embeddings.jsonl (id, date_year, embedding).")
    g.add_argument("--kind", choices=["journal", "conference"],
                   help="Автопути: {kind}/{kind}_test_embeddings.jsonl; модели: {kind}/models/v2; индексы: {kind}/2019..2023.jsonl.")

    p.add_argument("--models-dir", type=str,
                   help="Каталог с моделями (по умолчанию: {kind}/models/v2 или <queries>/../models/v2).")
    p.add_argument("--outdir", type=str,
                   help="Куда писать итоги (по умолчанию: {kind}/models/v2_result или рядом с models).")
    p.add_argument("--index-dir", type=str,
                   help="Где лежат 2019.jsonl..2023.jsonl (по умолчанию: {kind}/ или <queries>/..).")

    p.add_argument("--limit", type=int, default=50, help="Top-K (по умолчанию 50).")
    p.add_argument("--exclude-substr", type=str, default="0.8",
                   help="Исключить модели, если подстрока встречается в имени файла. Можно через запятую.")

    p.add_argument("--pool", choices=["max", "mean", "softmax", "topm"], default="softmax",
                   help="Агрегатор прототипов внутри источника (по умолчанию softmax).")
    p.add_argument("--tau", type=float, default=10.0, help="Температура для softmax.")
    p.add_argument("--m", type=int, default=3, help="m для topm (медленный).")
    p.add_argument("--use-priors", action="store_true", help="При softmax учитывать priors из модели, если есть.")

    p.add_argument("--query-batch", type=int, default=1024, help="Размер батча запросов (умерь если OOM).")
    p.add_argument("--count-lines", action="store_true", help="Точный tqdm по queries.")
    return p

# --------------------------- main ---------------------------

def main():
    args = build_argparser().parse_args()

    # определить пути
    if args.kind:
        root = Path(args.kind).resolve()
        queries_path = root / f"{args.kind}_test_embeddings.jsonl"
        models_dir = Path(args.models_dir).resolve() if args.models_dir else (root / "models" / "v2")
        outdir = Path(args.outdir).resolve() if args.outdir else (root / "models" / "v2_result")
        index_dir = Path(args.index_dir).resolve() if args.index_dir else root
        base_prefix = args.kind
    else:
        queries_path = Path(args.queries).resolve()
        base_prefix = queries_path.parent.name.split("_")[0] if "_" in queries_path.name else None
        models_dir = Path(args.models_dir).resolve() if args.models_dir else (queries_path.parent.parent / "models" / "v2")
        outdir = Path(args.outdir).resolve() if args.outdir else (models_dir.parent / "v2_result")
        index_dir = Path(args.index_dir).resolve() if args.index_dir else queries_path.parent

    if not queries_path.exists():
        print(f"[ERR] не найден файл запросов: {queries_path}", file=sys.stderr); sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    # собрать список моделей
    exclude_list = [s.strip() for s in (args.exclude_substr or "").split(",") if s.strip()]
    model_files = find_models(models_dir, base_prefix=base_prefix, exclude_substrings=exclude_list)
    if not model_files:
        print(f"[ERR] в {models_dir} не найдено подходящих мультивекторных моделей (исключения: {exclude_list})", file=sys.stderr)
        sys.exit(2)

    # выяснить размерность по первой строке queries
    q_total = count_lines(queries_path) if args.count_lines else None
    dim = None
    with queries_path.open("rb") as f:
        for raw in f:
            if not raw.strip(): continue
            rec = orjson.loads(raw)
            v = rec.get("embedding")
            if v is None: continue
            dim = int(np.asarray(v, dtype=np.float32).size); break
    if dim is None:
        print("[ERR] queries пустой или без embedding.", file=sys.stderr); sys.exit(3)

    # загрузить/построить множества допустимых source_id по годам
    year_src_sets = load_or_build_year_src_sets(index_dir)
    print(f"[*] year source sets: {sorted(year_src_sets.keys())}")

    # pack dir
    pack_dir = models_dir / "rankpack_multi"
    pack_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] models dir: {models_dir}")
    print(f"[*] packs dir:  {pack_dir}")
    print(f"[*] outdir:     {outdir}")
    print(f"[*] topK={args.limit}; pool={args.pool}"
          + (f"; tau={args.tau}" if args.pool == "softmax" else "")
          + (f"; m={args.m}" if args.pool == "topm" else "")
          + f"; query_batch={args.query_batch}")

    # цикл по моделям
    for mpath in model_files:
        # pack (если нет) → load
        pack_path = pack_multi_model(mpath, pack_dir, year_src_sets, expect_dim=dim)
        protos, groups, priors, src_ids, ptr_full, per_year = load_pack(pack_path)
        label = sanitize_model_label(mpath.stem, base_prefix or "")

        out_path = outdir / f"{mpath.stem}_top{args.limit}_{args.pool}.jsonl"
        wrote = 0

        # второй проход по queries: батчи по годам
        with out_path.open("wb") as fout:
            for yr, q_ids, Q in query_iter_grouped_by_year(queries_path,
                                                            total=q_total,
                                                            batch=args.query_batch):
                if yr not in per_year:
                    # для этой модели в этом году нет кандидатов
                    for qid in q_ids.tolist():
                        rec_out = {"model_file": str(mpath),
                                   "model": label,
                                   "pool": args.pool + (f"(tau={args.tau})" if args.pool == "softmax" else "") \
                                                    + (f"(m={int(args.m)})" if args.pool == "topm" else ""),
                                   "query_id": int(qid),
                                   "year_bucket": int(yr),
                                   "top": []}
                        fout.write(orjson.dumps(rec_out) + b"\n")
                        wrote += 1
                    continue

                perm_y, gidx_y, ptr_y = per_year[yr]
                if perm_y.size == 0 or gidx_y.size == 0:
                    for qid in q_ids.tolist():
                        rec_out = {"model_file": str(mpath),
                                   "model": label,
                                   "pool": args.pool + (f"(tau={args.tau})" if args.pool == "softmax" else "") \
                                                    + (f"(m={int(args.m)})" if args.pool == "topm" else ""),
                                   "query_id": int(qid),
                                   "year_bucket": int(yr),
                                   "top": []}
                        fout.write(orjson.dumps(rec_out) + b"\n")
                        wrote += 1
                    continue

                # матрица кандидатов года
                M_y  = protos[perm_y]            # [Py, D]
                pri_y = priors[perm_y]           # [Py]  (для softmax с priors)
                Gy  = gidx_y.size

                # косинусные схожести: [B, D] @ [D, Py] -> [B, Py]
                S_batch = Q @ M_y.T

                # агрегирование по группам
                if args.pool == "max":
                    agg = pool_max(S_batch, ptr_y)                     # [B, Gy]
                elif args.pool == "mean":
                    agg = pool_mean(S_batch, ptr_y)                    # [B, Gy]
                elif args.pool == "softmax":
                    agg = pool_softmax(S_batch, ptr_y, tau=float(args.tau),
                                       priors_cols=(pri_y if args.use_priors else None))
                else:  # topm
                    agg = pool_topm(S_batch, ptr_y, m=max(1, int(args.m)))

                # Top-K по источникам из gidx_y
                K = min(args.limit, Gy)
                idx = np.argpartition(-agg, K - 1, axis=1)[:, :K]     # [B, K]
                row = np.arange(agg.shape[0])[:, None]
                ord_in = np.argsort(-agg[row, idx], axis=1, kind="mergesort")
                idx = idx[row, ord_in]
                sc  = agg[row, idx]

                # local group idx -> глобальные source_id
                src_local  = gidx_y[idx]                               # [B, K]
                src_global = src_ids[src_local]                        # [B, K]

                # запись
                for i, qid in enumerate(q_ids.tolist()):
                    toplist = [
                        {"rank": r, "source_id": int(src_global[i, r-1]), "score": float(sc[i, r-1])}
                        for r in range(1, K+1)
                    ]
                    rec_out = {
                        "model_file": str(mpath),
                        "model": label,
                        "pool": args.pool + (f"(tau={args.tau})" if args.pool == "softmax" else "") \
                                        + (f"(m={int(args.m)})" if args.pool == "topm" else ""),
                        "query_id": int(qid),
                        "year_bucket": int(yr),
                        "top": toplist
                    }
                    fout.write(orjson.dumps(rec_out) + b"\n")
                    wrote += 1

        print(f"✔ {mpath.name} → {out_path}  ({wrote} запросов)")

    print("✓ done.")


if __name__ == "__main__":
    main()
