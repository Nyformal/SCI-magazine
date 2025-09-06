#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# nn_topk_by_year_fast.py
import argparse
import os
from pathlib import Path
from collections import defaultdict

import orjson
import numpy as np
from tqdm import tqdm

YEAR_BUCKETS = [2019, 2020, 2021, 2022, 2023]
EPS = 1e-8

def pick_bucket_year(y: int) -> int:
    if y <= 2019: return 2019
    if y == 2020: return 2020
    if y == 2021: return 2021
    if y == 2022: return 2022
    return 2023

def l2norm_rows(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    np.maximum(nrm, EPS, out=nrm)
    M = M / nrm
    return M.astype(np.float32, copy=False)

def ensure_cache_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_index_jsonl_to_arrays(year_file: Path):
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
        return np.asarray(ids), np.asarray(mat)

    # 2) читаем JSONL только один раз, нормируем, сохраняем кэш
    year_file = index_dir / f"{year}.jsonl"
    if not year_file.exists():
        raise FileNotFoundError(year_file)
    ids, mat = load_index_jsonl_to_arrays(year_file)
    mat = l2norm_rows(mat)
    save_npy_cache(cache_dir, year, ids, mat)
    return ids, mat

def topk_cosine_batched(ids: np.ndarray, mat_norm: np.ndarray, Q: np.ndarray, k: int) -> list[list[int]]:
    """
    ids: (N,), mat_norm: (N,D); Q: (B,D) — уже L2-нормированы.
    Возвращает список длины B: top-k id для каждой колонки.
    """
    # матричное умножение: [N,D] @ [D,B] = [N,B]
    S = mat_norm @ Q.T
    N, B = S.shape[0], Q.shape[0]
    k = min(k, N)

    results = []
    # выбираем топ-k для каждой колонки
    for j in range(B):
        col = S[:, j]
        if k == N:
            order = np.argsort(-col)
        else:
            part = np.argpartition(col, -k)[-k:]
            order = part[np.argsort(-col[part])]
        results.append(ids[order].tolist())
    return results

def parse_args():
    ap = argparse.ArgumentParser(
        description="Быстрый top-K по годовым индексам с кэшем .npy и батч-умножением."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--kind", choices=["journal", "conference"],
                   help="Возьмёт ./<kind>/<kind>_test_embeddings.jsonl")
    g.add_argument("--input", type=str,
                   help="Явный путь к JSONL с запросами (id, date_year, embedding).")

    ap.add_argument("--index-dir", type=str, default=None,
                    help="Где лежат 2019.jsonl..2023.jsonl (по умолчанию — рядом с входным файлом).")
    ap.add_argument("--k", type=int, default=1000, help="top-K (по умолчанию 1000).")
    ap.add_argument("--batch-size", type=int, default=512, help="Размер батча запросов (по умолчанию 512).")
    ap.add_argument("--out", type=str, default=None,
                    help="Куда писать результат (по умолчанию: *_top{k}.jsonl рядом с входным).")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="Каталог для .npy-кэша (по умолчанию: <index-dir>/.npy_cache).")
    ap.add_argument("--progress", action="store_true", help="Показывать tqdm.")
    ap.add_argument("--threads", type=int, default=None,
                    help="Принудительно выставить OMP/MKL_NUM_THREADS (лучше задавать до импорта NumPy).")
    return ap.parse_args()

def main():
    args = parse_args()

    # (не идеально — NumPy уже импортирован; лучше задавать переменные окружения ДО запуска скрипта)
    if args.threads:
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

    # 1) читаем ВСЕ запросы в память и группируем по бакету года
    queries_by_bucket = defaultdict(list)  # y -> list of (qid, vec)
    order = []                             # сохранить порядок (bucket,y_index,qid)
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
            # нормируем сразу
            n = float(np.linalg.norm(v))
            if n < EPS:
                continue
            v = (v / n).astype(np.float32, copy=False)
            qidx = len(queries_by_bucket[b])
            queries_by_bucket[b].append((int(qid), v))
            order.append((b, qidx, int(qid)))

    if not order:
        print("[WARN] Нет валидных запросов — нечего считать.")
        out_path.write_text("")
        return

    # 2) находим, какие индексы нужны, грузим КАЖДЫЙ ОДИН РАЗ (с .npy-кэшом)
    needed_years = sorted(queries_by_bucket.keys())
    index_cache = {}  # year -> (ids, mat_norm)
    for y in (tqdm(needed_years, desc="Load indices", unit="year") if args.progress else needed_years):
        ids, mat = load_index(y, index_dir, cache_dir)
        index_cache[y] = (ids, mat)

    # 3) обрабатываем по годам батчами и собираем ответы
    results_map = {}  # qid -> list[int]
    for y in needed_years:
        ids, mat = index_cache[y]
        bucket_qs = queries_by_bucket[y]
        # батчим
        B = args.batch_size
        for s in (tqdm(range(0, len(bucket_qs), B), desc=f"Search {y}", unit="batch") if args.progress else range(0, len(bucket_qs), B)):
            batch = bucket_qs[s:s+B]
            qids = [qid for qid, _ in batch]
            Q = np.vstack([vec for _, vec in batch]).astype(np.float32, copy=False)
            top_lists = topk_cosine_batched(ids, mat, Q, args.k)
            for qid, res in zip(qids, top_lists):
                results_map[qid] = res

    # 4) пишем результат в исходном порядке запросов
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
