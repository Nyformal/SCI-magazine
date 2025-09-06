#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import orjson
import numpy as np
from tqdm import tqdm


# ============================== utils ==============================

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x.astype(np.float32, copy=False)
    return (x / n).astype(np.float32, copy=False)


def power_iteration_first_pc(vectors_iter, dim: int, iters: int = 20, seed: int = 42) -> np.ndarray:
    """
    Power iteration для оценки первой главной компоненты.
    vectors_iter — итерируемое по векторам (или функция-генератор).
    """
    rng = np.random.default_rng(seed)
    v = l2norm(rng.normal(size=dim).astype(np.float32))
    for _ in range(iters):
        acc = np.zeros(dim, dtype=np.float32)
        it = vectors_iter() if callable(vectors_iter) else vectors_iter
        for x in it:
            acc += x * float(np.dot(x, v))
        v = l2norm(acc)
    return v


def robust_quantiles(X: np.ndarray, a: float):
    q_low = np.quantile(X, a, axis=0)
    q_hi  = np.quantile(X, 1.0 - a, axis=0)
    return q_low.astype(np.float32), q_hi.astype(np.float32)


def center(X: np.ndarray):
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    return (X - mu).astype(np.float32), mu


def pca_svd(Xc: np.ndarray, ncomp: int):
    full = min(ncomp, min(Xc.shape))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S = S[:full].astype(np.float32, copy=False)
    V = Vt[:full].astype(np.float32, copy=False)
    return S, V


def kmeans_l2(X: np.ndarray, k: int, iters: int = 30, seed: int = 42):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    if k > n:
        k = n
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        sims = X @ C.T         # косинус = dot (векторы нормированы)
        labels = sims.argmax(axis=1)
        newC = np.zeros_like(C)
        for j in range(k):
            mask = (labels == j)
            if not np.any(mask):
                newC[j] = X[rng.integers(0, n)]
            else:
                newC[j] = l2norm(X[mask].mean(axis=0))
        if np.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    return labels, C


# ========================= advanced aggregators =========================
# Все работают на L2-нормированных векторах X (n,d)

def agg_geom(X: np.ndarray, iters: int = 100, tol: float = 1e-6):
    if X.shape[0] == 1:
        return X[0]
    m = l2norm(X.mean(axis=0))
    eps = 1e-9
    for _ in range(iters):
        diff = X - m
        dist = np.linalg.norm(diff, axis=1) + eps
        inv = 1.0 / dist
        new_m = (X * inv[:, None]).sum(axis=0) / inv.sum()
        if np.linalg.norm(new_m - m) < tol:
            m = new_m
            break
        m = new_m
    return l2norm(m)


def agg_medoid(X: np.ndarray):
    if X.shape[0] == 1:
        return X[0]
    S = X @ X.T
    scores = S.mean(axis=1)
    return X[int(scores.argmax())]


def agg_tmean(X: np.ndarray, p: float = 0.8):
    n = X.shape[0]
    if n == 1:
        return X[0]
    c0 = l2norm(X.mean(axis=0))
    sims = X @ c0
    k = max(1, int(np.ceil(p * n)))
    idx = np.argsort(sims)[-k:]
    return l2norm(X[idx].mean(axis=0))


def agg_winsor(X: np.ndarray, a: float = 0.1):
    n, d = X.shape
    if n < 5 or not (0.0 < a < 0.5):
        return l2norm(X.mean(axis=0))
    low, hi = robust_quantiles(X, a)
    Xc = np.clip(X, low, hi)
    return l2norm(Xc.mean(axis=0))


def agg_kmeans_largest(X: np.ndarray, k: int = 4, seed: int = 42):
    n = X.shape[0]
    if n == 1 or k <= 1:
        return X[0]
    labels, C = kmeans_l2(X, k=k, seed=seed)
    sizes = np.bincount(labels, minlength=C.shape[0])
    j = int(sizes.argmax())
    return l2norm(C[j])


def agg_pcfuse(X: np.ndarray, ncomp: int = 2):
    if X.shape[0] == 1:
        return X[0]
    Xc, mu = center(X)
    ncomp = max(1, min(ncomp, min(X.shape)))
    S, V = pca_svd(Xc, ncomp)
    comp = (S[:, None] * V).sum(axis=0)
    return l2norm(comp)


def agg_svdshrink(X: np.ndarray, alpha: float = 0.1):
    if X.shape[0] == 1:
        return X[0]
    Xc, mu = center(X)
    r = min(X.shape)
    S, V = pca_svd(Xc, r)
    Sp = np.maximum(S - float(alpha), 0.0)
    if Sp.max() <= 0:
        return l2norm(V[0])
    comp = (Sp[:, None] * V).sum(axis=0)
    return l2norm(comp)


def agg_huber(X: np.ndarray, delta: float = 0.2, iters: int = 20):
    if X.shape[0] == 1:
        return X[0]
    m = l2norm(X.mean(axis=0))
    for _ in range(iters):
        sims = X @ m
        dist = 1.0 - sims
        w = np.ones_like(dist, dtype=np.float32)
        mask = dist > delta
        w[mask] = delta / (dist[mask] + 1e-9)
        mw = (X * w[:, None]).sum(axis=0) / (w.sum() + 1e-9)
        new_m = l2norm(mw)
        if np.linalg.norm(new_m - m) < 1e-6:
            m = new_m
            break
        m = new_m
    return m


def agg_attn2(X: np.ndarray, tau: float = 10.0):
    if X.shape[0] == 1:
        return X[0]
    c0 = l2norm(X.mean(axis=0))
    s = tau * (X @ c0)
    s -= float(s.max())
    w = np.exp(s).astype(np.float32)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return c0
    agg = (X * w[:, None]).sum(axis=0) / w_sum
    return l2norm(agg)


def agg_subtrim(X: np.ndarray, k: int = 3, p: float = 0.8):
    if X.shape[0] == 1:
        return X[0]
    Xc, mu = center(X)
    k = max(1, min(k, min(X.shape)))
    S, V = pca_svd(Xc, k)
    Z = Xc @ V.T
    mu_z = Z.mean(axis=0)
    nm = float(np.linalg.norm(mu_z))
    if nm < 1e-9:
        sims = -np.linalg.norm(Z, axis=1)
    else:
        sims = Z @ (mu_z / nm)
    m = max(1, int(np.ceil(p * X.shape[0])))
    idx = np.argsort(sims)[-m:]
    z_trim = Z[idx].mean(axis=0)
    a = z_trim @ V + mu
    return l2norm(a)


# ============================ CLI config ============================

BASIC_DEFAULT = ("mean", "sum", "max", "median", "cosmean", "rm1pc", "softmax")
ADV_DEFAULT = (
    "geom", "medoid", "tmean@0.8", "winsor@0.1",
    "kmeans@4", "pcfuse@2", "svdshrink@0.1",
    "huber@0.2", "attn2@10", "subtrim@3,0.8",
)
ALL_DEFAULT = BASIC_DEFAULT + ADV_DEFAULT


def parse_modes(s: str):
    """
    Разбор 'mean,geom,tmean@0.8,subtrim@3,0.8' -> [(name,[params...]),...]
    """
    out = []
    for tok in [t.strip().lower() for t in s.split(",") if t.strip()]:
        if "@" in tok:
            name, param = tok.split("@", 1)
            params = [p.strip() for p in param.split(",") if p.strip()]
        else:
            name, params = tok, []
        out.append((name, params))
    # снять дубли, сохранив порядок
    seen = set(); dedup = []
    for name, params in out:
        key = (name, tuple(params))
        if key not in seen:
            seen.add(key); dedup.append((name, params))
    return dedup


def parse_args():
    p = argparse.ArgumentParser(
        description="Единый агрегатор эмбеддингов по source_id (один вектор на source_id). Базовые и продвинутые режимы."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--kind", choices=["journal", "conference"],
                   help="Искать journal|conference _embeddings.jsonl в текущей папке.")
    g.add_argument("--input", type=str,
                   help="Явный путь к входному *.jsonl ({id, source_id, embedding}).")

    p.add_argument("--modes", type=str, default=",".join(ALL_DEFAULT),
                   help="Список режимов. "
                        "Базовые: mean,sum,max,median,cosmean,rm1pc,softmax. "
                        "Продвинутые: geom,medoid,tmean@p,winsor@a,kmeans@k,pcfuse@n,svdshrink@alpha,huber@delta,attn2@tau,subtrim@k,p")
    p.add_argument("--count-lines", action="store_true", help="Точный tqdm (медленнее).")
    p.add_argument("--strict", action="store_true", help="Падать на первой ошибке.")
    p.add_argument("--outdir", type=str, default=None, help="Каталог вывода (по умолчанию: ./models рядом с входным).")
    p.add_argument("--seed", type=int, default=42, help="Сид для kmeans.")
    p.add_argument("--tau", type=float, default=10.0, help="Температура для базового softmax-пуллинга.")
    return p.parse_args()


# ================================ main ================================

def main():
    args = parse_args()
    in_path = Path(args.input).resolve() if args.input else Path(f"{args.kind}_embeddings.jsonl").resolve()
    if not in_path.exists():
        print(f"[ERR] не найден входной файл: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.outdir).resolve() if args.outdir else (in_path.parent / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = in_path.stem
    if base.endswith("_embeddings"):
        base = base[: -len("_embeddings")]

    mode_specs = parse_modes(args.modes)
    # определим, какие блоки нужны
    need_median_raw = any(name == "median" for name, _ in mode_specs)
    need_softmax    = any(name == "softmax" for name, _ in mode_specs)
    need_cosmean    = any(name in ("cosmean", "rm1pc") for name, _ in mode_specs)
    need_adv_norm   = any(name in (
        "geom","medoid","tmean","winsor","kmeans","pcfuse","svdshrink","huber","attn2","subtrim"
    ) for name, _ in mode_specs)

    # накопители
    ids_map         = defaultdict(list)    # src -> [ids]
    sum_map         = {}                   # src -> sum(vec)
    max_map         = {}                   # src -> max(vec)
    count_map       = defaultdict(int)
    vectors_raw_map = defaultdict(list) if need_median_raw else None
    vecs_norm_map   = defaultdict(list) if (need_adv_norm or need_softmax or need_cosmean) else None
    global_sum_norm = np.zeros(0, dtype=np.float32) if need_softmax else None

    dim = None
    bad = 0
    total_lines = None
    if args.count_lines:
        with in_path.open("rb") as f:
            total_lines = sum(1 for _ in f)

    # чтение
    with in_path.open("rb") as f:
        for raw in tqdm(f, total=total_lines, unit="lines", desc="Reading"):
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                bad += 1
                if args.strict:
                    raise
                continue

            emb = rec.get("embedding")
            if emb is None:
                continue

            try:
                src = int(rec["source_id"])
                pid = int(rec["id"])
            except Exception:
                bad += 1
                if args.strict:
                    raise
                continue

            try:
                v = np.asarray(emb, dtype=np.float32)
            except Exception:
                bad += 1
                if args.strict:
                    raise
                continue

            if dim is None:
                dim = int(v.size)
                if need_softmax and global_sum_norm.size == 0:
                    global_sum_norm = np.zeros(dim, dtype=np.float32)
            elif v.size != dim:
                bad += 1
                if args.strict:
                    raise ValueError(f"Размерность не совпала: ожидалось {dim}, получено {v.size} (source_id={src})")
                continue

            ids_map[src].append(pid)
            count_map[src] += 1

            if src not in sum_map:
                sum_map[src] = v.copy()
            else:
                sum_map[src] += v

            if src not in max_map:
                max_map[src] = v.copy()
            else:
                np.maximum(max_map[src], v, out=max_map[src])

            if need_median_raw:
                vectors_raw_map[src].append(v)

            if (need_adv_norm or need_softmax or need_cosmean):
                vhat = l2norm(v)
                vecs_norm_map[src].append(vhat)
                if need_softmax:
                    global_sum_norm += vhat

    if dim is None:
        print("[ERR] Во входном нет валидных эмбеддингов.", file=sys.stderr)
        sys.exit(2)

    print(f"[OK] source_id: {len(ids_map)}; dim={dim}; плохих строк: {bad}")

    # подготовка для cosmean / rm1pc
    cos_mean_map = {}
    if need_cosmean:
        for src, lst in tqdm(vecs_norm_map.items(), total=len(vecs_norm_map), desc="cos-mean", unit="src"):
            s = np.sum(np.vstack(lst), axis=0).astype(np.float32, copy=False)
            cos_mean_map[src] = l2norm(s)

    pc1 = None
    if any(name == "rm1pc" for name, _ in mode_specs):
        pc1 = power_iteration_first_pc(cos_mean_map.values(), dim=dim, iters=20, seed=42)

    ghat = None
    if need_softmax:
        ghat = l2norm(global_sum_norm)

    # запись по режимам
    for name, params in mode_specs:
        out_path = out_dir / f"{base}_{name}.jsonl"
        wrote = 0
        it = tqdm(ids_map.items(), total=len(ids_map), desc=f"Writing {name}", unit="src")
        with out_path.open("wb") as fout:
            for src, pub_ids in it:
                # базовые режимы
                if name == "mean":
                    agg = (sum_map[src] / float(count_map[src])).astype(np.float32, copy=False)

                elif name == "sum":
                    agg = sum_map[src].astype(np.float32, copy=False)

                elif name == "max":
                    agg = max_map[src].astype(np.float32, copy=False)

                elif name == "median":
                    stack = np.vstack(vectors_raw_map[src])
                    agg = np.median(stack, axis=0).astype(np.float32, copy=False)

                elif name == "cosmean":
                    agg = cos_mean_map[src]

                elif name == "rm1pc":
                    v = cos_mean_map[src]
                    proj = float(np.dot(v, pc1))
                    agg = l2norm(v - pc1 * proj)

                elif name == "softmax":
                    Xhat = vecs_norm_map[src]
                    sims = np.array([float(np.dot(xh, ghat)) for xh in Xhat], dtype=np.float32)
                    s = args.tau * sims
                    s -= float(s.max())
                    w = np.exp(s).astype(np.float32)
                    w_sum = float(w.sum())
                    if w_sum <= 0.0:
                        agg = l2norm(np.mean(np.vstack(Xhat), axis=0).astype(np.float32))
                    else:
                        weighted = np.zeros(dim, dtype=np.float32)
                        for wi, xh in zip(w, Xhat):
                            weighted += wi * xh
                        agg = l2norm(weighted)

                # продвинутые режимы (все — на нормированных векторах)
                elif name == "geom":
                    agg = agg_geom(np.vstack(vecs_norm_map[src]))

                elif name == "medoid":
                    agg = agg_medoid(np.vstack(vecs_norm_map[src]))

                elif name == "tmean":
                    p = float(params[0]) if params else 0.8
                    p = min(max(p, 0.05), 1.0)
                    agg = agg_tmean(np.vstack(vecs_norm_map[src]), p=p)

                elif name == "winsor":
                    a = float(params[0]) if params else 0.1
                    a = min(max(a, 0.0), 0.49)
                    agg = agg_winsor(np.vstack(vecs_norm_map[src]), a=a)

                elif name == "kmeans":
                    k = int(round(float(params[0]))) if params else 4
                    k = max(2, k)
                    agg = agg_kmeans_largest(np.vstack(vecs_norm_map[src]), k=k, seed=args.seed)

                elif name == "pcfuse":
                    ncomp = int(round(float(params[0]))) if params else 2
                    ncomp = max(1, ncomp)
                    agg = agg_pcfuse(np.vstack(vecs_norm_map[src]), ncomp=ncomp)

                elif name == "svdshrink":
                    alpha = float(params[0]) if params else 0.1
                    agg = agg_svdshrink(np.vstack(vecs_norm_map[src]), alpha=alpha)

                elif name == "huber":
                    delta = float(params[0]) if params else 0.2
                    agg = agg_huber(np.vstack(vecs_norm_map[src]), delta=delta)

                elif name == "attn2":
                    tau = float(params[0]) if params else 10.0
                    agg = agg_attn2(np.vstack(vecs_norm_map[src]), tau=tau)

                elif name == "subtrim":
                    k = int(round(float(params[0]))) if len(params) >= 1 else 3
                    p = float(params[1]) if len(params) >= 2 else 0.8
                    k = max(1, k); p = min(max(p, 0.05), 1.0)
                    agg = agg_subtrim(np.vstack(vecs_norm_map[src]), k=k, p=p)

                else:
                    # неизвестный режим — пропустим
                    continue

                if agg.size != dim:
                    raise AssertionError(f"size mismatch for source_id={src}: {agg.size} != {dim}")

                rec = {
                    "source_id": int(src),
                    "ids": [int(x) for x in pub_ids],
                    "embedding": agg.tolist(),
                }
                fout.write(orjson.dumps(rec) + b"\n")
                wrote += 1

        print(f"✔ {name}: {wrote} записей → {out_path}")


if __name__ == "__main__":
    main()
