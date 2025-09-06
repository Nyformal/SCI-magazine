#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import orjson
import numpy as np
from tqdm import tqdm

YEAR_BUCKETS = [2019, 2020, 2021, 2022, 2023]


# ---------------- utils ----------------

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x.astype(np.float32, copy=False)
    return (x / n).astype(np.float32, copy=False)


def pick_bucket_year(y: int) -> int:
    if y <= 2019: return 2019
    if y == 2020: return 2020
    if y == 2021: return 2021
    if y == 2022: return 2022
    return 2023


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def should_skip(path: Path, exclude_substrings: List[str]) -> bool:
    low = path.name.lower()
    for s in exclude_substrings:
        s = s.strip().lower()
        if s and s in low:
            return True
    return False


def is_single_vector_model(path: Path, probe_lines: int = 10) -> bool:
    """Фильтруем только модели с полем 'embedding' (а не 'embeddings')."""
    try:
        with path.open("rb") as f:
            for _ in range(probe_lines):
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                if "embedding" in rec and "embeddings" not in rec:
                    return True
    except Exception:
        return False
    return False


# ------------- годовые source_id (кэш) -------------

def ensure_src_cache(index_dir: Path) -> Path:
    p = index_dir / ".src_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_year_source_ids(year_file: Path) -> np.ndarray:
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
            if sid is None:
                continue
            try:
                sids.add(int(sid))
            except Exception:
                pass
    return np.fromiter(sorted(sids), dtype=np.int64)


def load_or_build_year_src_sets(index_dir: Path) -> Dict[int, np.ndarray]:
    cache = ensure_src_cache(index_dir)
    out: Dict[int, np.ndarray] = {}
    any_found = False
    for y in YEAR_BUCKETS:
        jf = index_dir / f"{y}.jsonl"
        if not jf.exists():
            continue
        any_found = True
        npy = cache / f"{y}_source_ids.npy"
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
        raise RuntimeError(f"Нет годовых файлов 2019..2023 в {index_dir}")
    return out


# ------------- packing моделей в ранк-паки -------------

def pack_model_jsonl_to_npz(model_jsonl: Path,
                            npz_path: Path,
                            year_src_sets: Dict[int, np.ndarray],
                            expect_dim: int | None = None) -> None:
    """
    Упаковывает модель *.jsonl -> *.npz с:
      - mat (float32 L2-norm, отсортирована блоками по годам),
      - src_ids (int64) в том же порядке,
      - year_keys (int32) и year_starts (int64) — разметка блоков.
    """
    # 1) грузим модель (нормируем), собираем src_ids и вектора
    src_ids: List[int] = []
    vecs: List[np.ndarray] = []

    total = count_lines(model_jsonl)
    with model_jsonl.open("rb") as f:
        for raw in tqdm(f, total=total, unit="lines", desc=f"Pack load {model_jsonl.name}"):
            if not raw.strip():
                continue
            rec = orjson.loads(raw)
            if "embedding" not in rec or "source_id" not in rec:
                continue
            v = np.asarray(rec["embedding"], dtype=np.float32)
            if expect_dim is not None and v.size != expect_dim:
                continue
            src_ids.append(int(rec["source_id"]))
            vecs.append(l2norm(v))

    if not vecs:
        raise RuntimeError(f"Модель пуста или несовместима по размерности: {model_jsonl}")

    src_ids = np.asarray(src_ids, dtype=np.int64)
    mat = np.vstack(vecs).astype(np.float32, copy=False)

    # 2) вычисляем индексы строк для каждого года
    id2row = {int(s): i for i, s in enumerate(src_ids.tolist())}
    blocks: List[np.ndarray] = []
    year_keys: List[int] = []
    starts: List[int] = [0]

    for y in YEAR_BUCKETS:
        allowed = year_src_sets.get(y)
        if allowed is None or allowed.size == 0:
            blocks.append(np.empty(0, dtype=np.int64))
            starts.append(starts[-1])
            year_keys.append(y)
            continue
        idx = [id2row[s] for s in allowed.tolist() if s in id2row]
        idx_sorted = np.asarray(sorted(idx), dtype=np.int64)
        blocks.append(idx_sorted)
        starts.append(starts[-1] + idx_sorted.size)
        year_keys.append(y)

    # 3) формируем permutation: просто конкатенируем блоки (год за годом)
    perm = np.concatenate([b for b in blocks if b.size > 0], axis=0) if any(b.size > 0 for b in blocks) else np.empty(0, dtype=np.int64)

    # Если какие-то строки не попали ни в один год — можно добавить их в хвост (не используются, но порядок полный)
    covered = np.zeros(src_ids.size, dtype=bool)
    covered[perm] = True
    rest = np.where(~covered)[0]
    if rest.size > 0:
        perm = np.concatenate([perm, rest], axis=0)

    src_ids_sorted = src_ids[perm]
    mat_sorted = mat[perm]

    # Пересчитаем starts под фактический perm (только по годам), хвост не нужен в годах
    # starts[i] — начало блока года i в конкатенации; нужно собрать заново из sizes реальных blocks
    sizes = [b.size for b in blocks]
    year_starts = [0]
    acc = 0
    for sz in sizes:
        year_starts.append(acc + sz)
        acc += sz
    year_starts = np.asarray(year_starts, dtype=np.int64)
    year_keys = np.asarray(year_keys, dtype=np.int32)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path,
             src_ids=src_ids_sorted,
             mat=mat_sorted,
             year_keys=year_keys,
             year_starts=year_starts)


def load_rankpack(npz_path: Path):
    """
    Загружает rankpack (mmap):
      returns dict with keys: 'src_ids', 'mat', 'year_keys', 'year_starts'
    """
    z = np.load(npz_path, allow_pickle=False, mmap_mode="r")
    return {
        "src_ids": z["src_ids"],
        "mat": z["mat"],
        "year_keys": z["year_keys"],
        "year_starts": z["year_starts"],
    }


# ------------- загрузка запросов -------------

def load_queries_grouped(queries_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Читает *_test_embeddings.jsonl, группирует по бакетам лет.
    Возвращает: {year: {"ids": np.int64 [Qy], "mat": float32 [Qy,d] (L2norm)}}
    """
    ids_by: Dict[int, List[int]] = {y: [] for y in YEAR_BUCKETS}
    vecs_by: Dict[int, List[np.ndarray]] = {y: [] for y in YEAR_BUCKETS}
    dim = None

    total = count_lines(queries_path)
    with queries_path.open("rb") as f:
        for raw in tqdm(f, total=total, unit="q", desc=f"Load queries {queries_path.name}"):
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                continue
            qid = rec.get("id")
            emb = rec.get("embedding")
            y   = rec.get("date_year")
            if qid is None or emb is None or y is None:
                continue
            v = np.asarray(emb, dtype=np.float32)
            if dim is None:
                dim = v.size
            elif v.size != dim:
                # пропускаем битые
                continue
            ids_by[pick_bucket_year(int(y))].append(int(qid))
            vecs_by[pick_bucket_year(int(y))].append(l2norm(v))

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for y in YEAR_BUCKETS:
        if not ids_by[y]:
            out[y] = {"ids": np.empty(0, dtype=np.int64), "mat": np.empty((0, dim or 0), dtype=np.float32)}
        else:
            out[y] = {
                "ids": np.asarray(ids_by[y], dtype=np.int64),
                "mat": np.vstack(vecs_by[y]).astype(np.float32, copy=False)
            }
    return out


# ------------- batched top-K по блоку кандидатов -------------

def batched_topk_against_block(Q: np.ndarray,
                               M: np.ndarray,
                               K: int,
                               cand_chunk: int,
                               query_batch: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q: [B, d] — L2norm
    M: [N, d] — L2norm (кандидаты ОДНОГО годового блока, contiguous!)
    Возвращает:
      top_idx:   [B, K] индексы относительно M (0..N-1)
      top_score: [B, K] косинус-оценки
    """
    B, d = Q.shape
    N, d2 = M.shape
    assert d == d2

    K = min(K, N)
    if K <= 0 or B == 0 or N == 0:
        return (np.empty((B, 0), dtype=np.int64),
                np.empty((B, 0), dtype=np.float32))

    # Разобьём кандидатов на чанки по столбцам
    chunks = [(s, min(s + cand_chunk, N)) for s in range(0, N, cand_chunk)]

    all_top_idx = []
    all_top_scr = []

    for qs in range(0, B, query_batch):
        qe = min(qs + query_batch, B)
        Qb = Q[qs:qe]                   # [b, d]
        b = Qb.shape[0]
        if b == 0:
            continue

        # Текущий top-K (инициализация -inf и -1)
        top_scores = np.full((b, K), -np.inf, dtype=np.float32)
        top_index  = np.full((b, K), -1,     dtype=np.int64)

        row_idx = np.arange(b)[:, None]  # [b,1] — пригодится для батчевого индексирования

        for s, e in chunks:
            Mc = M[s:e]                  # [c, d]
            c = Mc.shape[0]
            if c == 0:
                continue

            sims = Qb @ Mc.T             # [b, c]

            # Смерджим старый top-K и новые оценки: [b, K+c]
            merged = np.concatenate([top_scores, sims], axis=1)

            # Берём top-K позиций по строкам
            # argpartition по возрастанию, поэтому выделяем правую часть
            part = np.argpartition(merged, kth=merged.shape[1]-K, axis=1)[:, -K:]        # [b, K]
            part_vals = merged[row_idx, part]                                           # [b, K]
            order = np.argsort(-part_vals, axis=1)                                      # [b, K]
            best_cols = part[row_idx, order]                                            # [b, K] — лучшие столбцы в merged

            # Новые значения скороров
            new_scores = merged[row_idx, best_cols]                                     # [b, K]

            # Формируем новые индексы кандидатов в M.
            # best_cols < K  -> берём из старого top_index
            # best_cols >= K -> это новые из sims; глобальный индекс = s + (best_cols - K)
            is_new   = (best_cols >= K)
            offset   = best_cols - K               # для старых будет отрицательный — далее замаскируем

            # Для старых: безопасный gather
            safe_old_cols = np.where(is_new, 0, best_cols)   # где новое — просто 0 (валидный столбец)
            gathered_old  = np.take_along_axis(top_index, safe_old_cols, axis=1)  # [b, K]

            # Для новых: s + offset (offset валиден только там, где is_new=True)
            gathered_new = s + np.maximum(offset, 0)

            new_index = np.where(is_new, gathered_new, gathered_old).astype(np.int64)   # [b, K]

            # Обновляем текущий top-K
            top_scores = new_scores.astype(np.float32, copy=False)
            top_index  = new_index

        all_top_idx.append(top_index)
        all_top_scr.append(top_scores)

    return np.vstack(all_top_idx), np.vstack(all_top_scr)


# ------------- поиск по всем моделям -------------

def sanitize_model_label(stem: str, base_prefix: str) -> str:
    return stem[len(base_prefix) + 1:] if base_prefix and stem.startswith(base_prefix + "_") else stem


def find_models(models_dir: Path, base_prefix: str | None, exclude_substrings: List[str]) -> List[Path]:
    if not models_dir.exists():
        return []
    files = sorted(p for p in models_dir.glob("*.jsonl") if is_single_vector_model(p))
    if base_prefix:
        files = sorted(files, key=lambda p: (0 if p.stem.startswith(base_prefix + "_") else 1, p.name))
    return [p for p in files if not should_skip(p, exclude_substrings)]


# ---------------- CLI ----------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Быстрый top-K по одновекторным моделям с годовым фильтром: ранк-паки + батч-GEMM."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--kind", choices=["journal", "conference"],
                   help="Автопути: {kind}/{kind}_test_embeddings.jsonl, модели — {kind}/models/v1.")
    g.add_argument("--queries", type=str,
                   help="Явный путь к *_test_embeddings.jsonl.")

    p.add_argument("--models-dir", type=str,
                   help="Каталог моделей (*.jsonl). По умолчанию: {kind}/models/v1 или рядом с queries.")
    p.add_argument("--packs-dir", type=str,
                   help="Куда класть *.npz ранк-паки. По умолчанию: <models-dir>/rankpack")
    p.add_argument("--index-dir", type=str,
                   help="Где лежат 2019.jsonл..2023.jsonл. По умолчанию: {kind}/ или рядом с queries.")
    p.add_argument("--outdir", type=str,
                   help="Куда писать результаты. По умолчанию: {kind}/models/v1_result")

    p.add_argument("--limit", type=int, default=50, help="Top-K (default 50)")
    p.add_argument("--exclude-substr", type=str, default="0.8", help="Исключить модели по подстрокам (через запятую).")

    # Параметры производительности:
    p.add_argument("--prepare-only", action="store_true", help="Только упаковать модели и выйти.")
    p.add_argument("--query-batch", type=int, default=2048, help="Размер батча запросов (default 2048).")
    p.add_argument("--cand-chunk", type=int, default=16384, help="Чанк кандидатов по столбцам (default 16384).")
    p.add_argument("--count-lines", action="store_true", help="Точный tqdm на queries.")
    return p


# ---------------- main ----------------

def main():
    args = build_argparser().parse_args()

    # пути
    if args.kind:
        root = Path(args.kind).resolve()
        queries_path = root / f"{args.kind}_test_embeddings.jsonl"
        models_dir = Path(args.models_dir).resolve() if args.models_dir else (root / "models" / "v1")
        packs_dir = Path(args.packs_dir).resolve() if args.packs_dir else (models_dir / "rankpack")
        index_dir = Path(args.index_dir).resolve() if args.index_dir else root
        outdir = Path(args.outdir).resolve() if args.outdir else (models_dir.parent / "v1_result")
        base_prefix = args.kind
    else:
        queries_path = Path(args.queries).resolve()
        base_prefix = queries_path.parent.name.split("_")[0] if "_" in queries_path.name else None
        models_dir = Path(args.models_dir).resolve() if args.models_dir else (queries_path.parent.parent / "models" / "v1")
        packs_dir = Path(args.packs_dir).resolve() if args.packs_dir else (models_dir / "rankpack")
        index_dir = Path(args.index_dir).resolve() if args.index_dir else queries_path.parent.parent
        outdir = Path(args.outdir).resolve() if args.outdir else (models_dir.parent / "v1_result")

    if not queries_path.exists():
        print(f"[ERR] queries not found: {queries_path}", file=sys.stderr)
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)
    packs_dir.mkdir(parents=True, exist_ok=True)

    # годовые множества source_id
    year_src_sets = load_or_build_year_src_sets(index_dir)
    print(f"[*] year source sets: {sorted(year_src_sets.keys())}")

    # список моделей
    exclude = [s.strip() for s in (args.exclude_substr or "").split(",") if s.strip()]
    model_files = find_models(models_dir, base_prefix=base_prefix, exclude_substrings=exclude)
    if not model_files:
        print(f"[ERR] no models in {models_dir} (exclude={exclude})", file=sys.stderr)
        sys.exit(2)

    # выяснить dim по первой строке queries
    first_dim = None
    with queries_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
                emb = rec.get("embedding")
                if emb is None:
                    continue
                first_dim = int(np.asarray(emb, dtype=np.float32).size)
                break
            except Exception:
                continue
    if first_dim is None:
        print("[ERR] queries file has no embeddings.", file=sys.stderr)
        sys.exit(3)

    # упаковка моделей (если нет .npz)
    for mpath in model_files:
        pack_path = packs_dir / (mpath.stem + ".npz")
        if pack_path.exists():
            continue
        print(f"[*] packing {mpath.name} -> {pack_path.name}")
        pack_model_jsonl_to_npz(mpath, pack_path, year_src_sets, expect_dim=first_dim)

    if args.prepare_only:
        print("✓ prepare-only: done.")
        return

    # запрашиваем и группируем все queries (разово)
    grouped = load_queries_grouped(queries_path)
    for y in YEAR_BUCKETS:
        qi = grouped[y]["ids"].shape[0]
        print(f"  year {y}: {qi} queries")

    print(f"[*] models dir: {models_dir}")
    print(f"[*] packs dir:  {packs_dir}")
    print(f"[*] outdir:     {outdir}")
    print(f"[*] topK={args.limit}; query_batch={args.query_batch}; cand_chunk={args.cand_chunk}")

    # ранжирование по моделям
    for mpath in model_files:
        pack_path = packs_dir / (mpath.stem + ".npz")
        if not pack_path.exists():
            print(f"[WARN] pack missing for {mpath.name}, skipping.", file=sys.stderr)
            continue

        pack = load_rankpack(pack_path)
        src_ids_sorted = pack["src_ids"]      # mmap
        mat_sorted = pack["mat"]              # mmap
        year_keys = pack["year_keys"]
        year_starts = pack["year_starts"]

        model_label = sanitize_model_label(mpath.stem, base_prefix or "")
        out_path = outdir / f"{mpath.stem}_top{args.limit}.jsonl"
        wrote = 0

        with out_path.open("wb") as fout:
            for y in YEAR_BUCKETS:
                # интервал кандидатов для этого года
                # year_starts: длина = len(Y)+1
                # индекс года по порядку:
                try:
                    yi = int(np.where(year_keys == y)[0][0])
                except Exception:
                    # год отсутствует в паке
                    continue
                s, e = int(year_starts[yi]), int(year_starts[yi + 1])
                if e <= s:
                    continue

                # запросы этого года
                Q_ids = grouped[y]["ids"]
                Q_mat = grouped[y]["mat"]
                if Q_mat.shape[0] == 0:
                    continue

                # считаем топK батчами
                top_idx, top_scr = batched_topk_against_block(
                    Q=Q_mat,
                    M=mat_sorted[s:e],
                    K=args.limit,
                    cand_chunk=max(1024, int(args.cand_chunk)),
                    query_batch=max(128, int(args.query_batch)),
                )
                # map индексы -> source_id
                src_slice = src_ids_sorted[s:e]
                top_src = src_slice[top_idx]    # [B,K]

                # пишем построчно (соблюдём прежний формат)
                for qid, row_src, row_scr in zip(Q_ids.tolist(), top_src.tolist(), top_scr.tolist()):
                    top = [
                        {"rank": rnk, "source_id": int(sid), "score": float(sc)}
                        for rnk, (sid, sc) in enumerate(zip(row_src, row_scr), start=1)
                    ]
                    rec = {
                        "model_file": str(mpath),
                        "model": model_label,
                        "query_id": int(qid),
                        "year_bucket": int(y),
                        "top": top
                    }
                    fout.write(orjson.dumps(rec) + b"\n")
                    wrote += 1

        print(f"✔ {mpath.name}: {wrote} запросов → {out_path}")


if __name__ == "__main__":
    main()
