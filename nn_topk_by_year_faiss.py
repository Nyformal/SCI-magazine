import argparse
import os
from pathlib import Path
from collections import defaultdict

import orjson
import numpy as np
from tqdm import tqdm

# попытка импорта FAISS CPU/GPU
try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("FAISS не установлен. Установи faiss-cpu или faiss-gpu.") from e

YEAR_BUCKETS = [2019, 2020, 2021, 2022, 2023]
EPS = 1e-8


def pick_bucket_year(y: int) -> int:
    if y <= 2019: return 2019
    if y == 2020: return 2020
    if y == 2021: return 2021
    if y == 2022: return 2022
    return 2023


def l2norm_rows(M: np.ndarray) -> np.ndarray:
    """L2-нормировка по строкам."""
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    np.maximum(nrm, EPS, out=nrm)
    M = M / nrm
    return M.astype(np.float32, copy=False)


def ensure_cache_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_index_jsonl_to_arrays(year_file: Path):
    """Читает <year>.jsonl -> ids(np.int64), mat(np.float32 [N,D])."""
    ids, rows = [], []
    dim = None
    with year_file.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = orjson.loads(raw)
            emb = rec.get("embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32)
            if dim is None:
                dim = v.size
            elif v.size != dim:
                # пропустим битую строку
                continue
            ids.append(int(rec["id"]))
            rows.append(v)
    if not rows:
        raise RuntimeError(f"Пустой индекс: {year_file}")
    mat = np.vstack(rows).astype(np.float32, copy=False)
    ids = np.asarray(ids, dtype=np.int64)
    return ids, mat


def save_npy_cache(cache_dir: Path, year: int, ids: np.ndarray, mat_norm: np.ndarray):
    np.save(cache_dir / f"{year}.ids.npy", ids)
    np.save(cache_dir / f"{year}.mat.npy", mat_norm)


def load_npy_cache(cache_dir: Path, year: int):
    ids_p = cache_dir / f"{year}.ids.npy"
    mat_p = cache_dir / f"{year}.mat.npy"
    if ids_p.exists() and mat_p.exists():
        ids = np.load(ids_p, mmap_mode="r")
        mat = np.load(mat_p, mmap_mode="r")
        return ids, mat
    return None, None


def load_index(year: int, index_dir: Path, cache_dir: Path):
    # 1) пробуем .npy кэш
    ids, mat = load_npy_cache(cache_dir, year)
    if ids is not None and mat is not None:
        # приведём к обычным ndarray (mmap остаётся, но faiss всё равно скопирует в свою память)
        return np.asarray(ids), np.asarray(mat)

    # 2) читаем JSONL, нормируем, сохраняем кэш
    year_file = index_dir / f"{year}.jsonl"
    if not year_file.exists():
        raise FileNotFoundError(year_file)
    ids, mat = load_index_jsonl_to_arrays(year_file)
    mat = l2norm_rows(mat)
    save_npy_cache(cache_dir, year, ids, mat)
    return ids, mat


def build_faiss_index_ip(mat_norm: np.ndarray, use_gpu: bool = False):
    """
    Строит FAISS IndexFlatIP (IP по нормированным = cosine).
    Если use_gpu=True и доступен faiss-gpu, переносит на GPU.
    """
    d = int(mat_norm.shape[1])
    index = faiss.IndexFlatIP(d)
    # На всякий случай убедимся в памяти и в C-контигуозности
    xb = np.ascontiguousarray(mat_norm, dtype=np.float32)

    if use_gpu:
        # если faiss-gpu установлен, можно использовать все GPU
        try:
            res = faiss.StandardGpuResources()
            # по-дефолту на все GPU:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            # если нет faiss-gpu — тихо продолжаем на CPU
            pass

    index.add(xb)
    return index


def search_faiss(index, ids: np.ndarray, Q: np.ndarray, k: int, batch_size: int, show_prog: bool = False):
    """
    Поиск батчами: Q[B,D] -> список результатов для каждой строки (список списков id).
    """
    k = min(k, ids.size)
    results = []
    rng = range(0, Q.shape[0], batch_size)
    if show_prog:
        rng = tqdm(rng, desc="FAISS search", unit="batch")
    for s in rng:
        e = min(s + batch_size, Q.shape[0])
        D, I = index.search(Q[s:e], k)   # I: локальные индексы базовых векторов
        # маппим в глобальные id
        for irow in range(I.shape[0]):
            res_ids = ids[I[irow]].tolist()
            results.append(res_ids)
    return results


def parse_args():
    ap = argparse.ArgumentParser(
        description="FAISS top-K по годовым индексам (2019..2023). Косинус через IP на L2-нормированных."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--kind", choices=["journal", "conference"],
                   help="Возьмёт ./<kind>/<kind>_test_embeddings.jsonl")
    g.add_argument("--input", type=str,
                   help="Явный путь к JSONL с запросами (id, date_year, embedding).")

    ap.add_argument("--index-dir", type=str, default=None,
                    help="Где лежат 2019.jsonl..2023.jsonl (по умолчанию — рядом с входным файлом).")
    ap.add_argument("--k", type=int, default=1000, help="top-K (по умолчанию 1000).")
    ap.add_argument("--batch-size", type=int, default=1024, help="Размер батча запросов (по умолчанию 1024).")
    ap.add_argument("--out", type=str, default=None,
                    help="Куда писать результат (по умолчанию: *_top{k}.jsonl рядом с входным).")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="Каталог для .npy-кэша (по умолчанию: <index-dir>/.npy_cache).")
    ap.add_argument("--use-gpu", action="store_true",
                    help="Использовать faiss-gpu (если установлен).")
    ap.add_argument("--progress", action="store_true", help="Показывать tqdm.")
    ap.add_argument("--threads", type=int, default=None,
                    help="faiss.omp_set_num_threads / BLAS threads; можно оставить по умолчанию.")
    return ap.parse_args()


def main():
    args = parse_args()

    # потоки
    if args.threads and args.threads > 0:
        try:
            faiss.omp_set_num_threads(args.threads)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    # входные пути
    if args.input:
        q_path = Path(args.input).resolve()
        base_dir = q_path.parent
        base_name = q_path.stem
    else:
        base_dir = Path.cwd() / args.kind
        q_path = base_dir / f"{args.kind}_test_embeddings.jsonl"
        base_name = q_path.stem

    if not q_path.exists():
        raise FileNotFoundError(q_path)

    index_dir = Path(args.index_dir).resolve() if args.index_dir else base_dir

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = base_dir / f"{base_name.rsplit('_embeddings',1)[0]}_top{args.k}.jsonl"

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (index_dir / ".npy_cache")
    ensure_cache_dir(cache_dir)

    # 1) читаем все запросы, группируем по году, нормируем
    queries_by_bucket = defaultdict(list)  # y -> list of (qid, vec)
    order = []                             # чтобы сохранить исходный порядок вывода
    with q_path.open("rb") as f:
        it = tqdm(f, desc=f"Read queries {q_path.name}", unit="lines") if args.progress else f
        for raw in it:
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                continue
            emb = rec.get("embedding"); qid = rec.get("id"); y = rec.get("date_year")
            if emb is None or qid is None or y is None:
                continue
            b = pick_bucket_year(int(y))
            v = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n < EPS:
                continue
            v = (v / n).astype(np.float32, copy=False)
            qidx = len(queries_by_bucket[b])
            queries_by_bucket[b].append((int(qid), v))
            order.append((b, qidx, int(qid)))

    if not order:
        print("[WARN] Нет валидных запросов — нечего искать.")
        out_path.write_text("")
        return

    # 2) загрузим нужные индексы (по годам) + соберём FAISS индексы
    needed_years = sorted(queries_by_bucket.keys())
    base_cache = {}   # year -> (ids, mat_norm)
    faiss_index = {}  # year -> faiss index

    for y in (tqdm(needed_years, desc="Load base & build FAISS", unit="year") if args.progress else needed_years):
        ids, mat = load_index(y, index_dir, cache_dir)  # нормированная матрица
        base_cache[y] = (ids, mat)
        # строим FAISS индекс на нужной базе
        faiss_index[y] = build_faiss_index_ip(mat, use_gpu=args.use_gpu)

    # 3) обработаем по годам батчами
    results_map = {}  # qid -> [ids...]
    for y in needed_years:
        ids, mat = base_cache[y]
        idx = faiss_index[y]

        bucket_qs = queries_by_bucket[y]
        # батчим
        B = max(1, int(args.batch_size))
        ranges = range(0, len(bucket_qs), B)
        if args.progress:
            ranges = tqdm(ranges, desc=f"Search {y}", unit="batch")

        for s in ranges:
            batch = bucket_qs[s:s+B]
            qids = [qid for qid, _ in batch]
            Q = np.vstack([vec for _, vec in batch]).astype(np.float32, copy=False)
            # top-k
            res_lists = search_faiss(idx, ids, Q, k=args.k, batch_size=Q.shape[0], show_prog=False)
            for qid, res in zip(qids, res_lists):
                results_map[qid] = res

    # 4) пишем в исходном порядке
    wrote = 0
    with out_path.open("wb") as fout:
        for _, _, qid in order:
            res = results_map.get(qid)
            if res is None:
                continue
            fout.write(orjson.dumps({"id": int(qid), "results": res}) + b"\n")
            wrote += 1

    print(f"✔ готово: {out_path} (строк: {wrote})")


if __name__ == "__main__":
    main()
