#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import orjson
import numpy as np
from tqdm import tqdm


# ============================ вспомогательные ============================

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x.astype(np.float32, copy=False)
    return (x / n).astype(np.float32, copy=False)


def parse_modes(spec: str):
    """
    'kmeans@8,fps@6,pca@5,softk@6,12,10,byyear@3' ->
    [('kmeans',['8']), ('fps',['6']), ('pca',['5']), ('softk',['6','12','10']), ('byyear',['3'])]
    """
    out, seen = [], set()
    for tok in [t.strip().lower() for t in spec.split(",") if t.strip()]:
        if "@" in tok:
            name, params = tok.split("@", 1)
            plist = [p.strip() for p in params.split(",") if p.strip()]
        else:
            name, plist = tok, []
        key = (name, tuple(plist))
        if key not in seen:
            seen.add(key); out.append((name, plist))
    return out


def kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator):
    """
    Инициализация kmeans++ для сферического k-means (X — L2-нормированы).
    Возвращает индексы выбранных центров.
    """
    n = X.shape[0]
    idxs = [int(rng.integers(0, n))]
    # расстояние ~ (1 - dot), отслеживаем минимальные
    dmin = 1.0 - (X @ X[idxs[0]].T)
    for _ in range(1, k):
        # вероятность ∝ dmin^2
        w = np.maximum(dmin, 0.0)
        w = w * w
        s = float(w.sum())
        if s <= 0:
            idxs.append(int(rng.integers(0, n)))
            continue
        probs = w / s
        next_idx = int(rng.choice(n, p=probs))
        idxs.append(next_idx)
        dmin = np.minimum(dmin, 1.0 - (X @ X[next_idx].T))
    return np.array(idxs, dtype=np.int64)


def spherical_kmeans(X: np.ndarray, k: int, iters: int = 30, seed: int = 42):
    """
    Сферический k-means: косинусная близость = dot, X — уже L2-нормированы.
    Возвращает labels (n,), centers (k,d).
    """
    n, d = X.shape
    k = min(max(1, k), n)
    rng = np.random.default_rng(seed)
    centers = X[kmeanspp_init(X, k, rng)].copy()

    for _ in range(iters):
        sims = X @ centers.T           # (n,k)
        labels = sims.argmax(axis=1)
        newC = np.zeros_like(centers)
        for j in range(k):
            mask = (labels == j)
            if not np.any(mask):
                newC[j] = X[int(rng.integers(0, n))]
            else:
                newC[j] = l2norm(X[mask].mean(axis=0))
        if np.allclose(newC, centers, atol=1e-5):
            centers = newC
            break
        centers = newC
    # финальные метки
    labels = (X @ centers.T).argmax(axis=1)
    return labels, centers


def furthest_point_sampling(X: np.ndarray, m: int, seed: int = 42):
    """
    FPS для косинуса (на L2-нормированных X). Возвращает индексы выбранных точек.
    """
    n = X.shape[0]
    m = min(max(1, m), n)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(0, n))]
    # минимальная "дистанция" = 1 - dot до ближайшего выбранного
    dmin = 1.0 - (X @ X[chosen[0]].T)
    for _ in range(1, m):
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, 1.0 - (X @ X[i].T))
    return np.array(chosen, dtype=np.int64)


def pca_components(X: np.ndarray, r: int):
    """
    PCA по центрированным нормированным векторам, возвращает r компонент (юнит-векторы) и сингулярные значения.
    """
    r = max(1, min(r, min(X.shape)))
    mu = X.mean(axis=0)
    Xc = (X - mu).astype(np.float32)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:r].astype(np.float32)
    comps = np.vstack([l2norm(c) for c in comps])
    Sv = S[:r].astype(np.float32)
    return comps, Sv


def soft_kmeans(X: np.ndarray, k: int, tau: float = 10.0, iters: int = 10, seed: int = 42):
    """
    «Мягкий» сферический k-means: веса = softmax(tau * dot(x, c_j)).
    Возвращает centers (k,d), weights (k,) — доли кластера.
    """
    n, d = X.shape
    k = min(max(1, k), n)
    rng = np.random.default_rng(seed)
    _, centers = spherical_kmeans(X, k=k, iters=10, seed=seed)

    for _ in range(iters):
        logits = X @ centers.T               # (n,k)
        logits *= float(tau)
        logits -= logits.max(axis=1, keepdims=True)
        W = np.exp(logits).astype(np.float32)
        Z = W.sum(axis=1, keepdims=True) + 1e-9
        P = W / Z                            # (n,k) soft-assign
        denom = P.sum(axis=0) + 1e-9         # (k,)
        newC = (P.T @ X) / denom[:, None]    # (k,d)
        for j in range(k):
            newC[j] = l2norm(newC[j])
        if np.allclose(newC, centers, atol=1e-5):
            centers = newC
            break
        centers = newC

    weights = (P.sum(axis=0) / float(n)).astype(np.float32)
    return centers, weights


# ================================ CLI ================================

DEFAULT_MULTI = "kmeans@4,fps@4,pca@3,cmedoids@4,softk@4,10,10,byyear@5"

def build_argparser():
    p = argparse.ArgumentParser(
        description="Мультивекторная агрегация эмбеддингов по source_id (несколько прототипов на источник)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--kind", choices=["journal", "conference"],
                   help="Искать {kind}_embeddings.jsonl в текущей папке.")
    g.add_argument("--input", type=str,
                   help="Явный путь к входному *.jsonl (поля: id, source_id, embedding[, date_year]).")

    p.add_argument("--modes", type=str, default=DEFAULT_MULTI,
                   help=("Список режимов через запятую. Доступно: "
                         "kmeans@k, cmedoids@k, fps@m, softk@k,tau,iters, pca@r, byyear@minN. "
                         f"По умолчанию: {DEFAULT_MULTI}"))

    p.add_argument("--outdir", type=str, default=None,
                   help="Каталог вывода (по умолчанию: ./models рядом с входным).")
    p.add_argument("--count-lines", action="store_true",
                   help="Точный tqdm (медленнее).")
    p.add_argument("--seed", type=int, default=42, help="Сид для инициализаций.")
    return p


# ================================ main ================================

def main():
    args = build_argparser().parse_args()

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

    # Что нам нужно копить:
    need_years = any(name == "byyear" for name, _ in mode_specs)

    vecs_map   = defaultdict(list)   # src -> [vec ...]  (float32)
    ids_map    = defaultdict(list)   # src -> [id ...]
    years_map  = defaultdict(list) if need_years else None

    dim = None
    bad = 0
    total_lines = None
    if args.count_lines:
        with in_path.open("rb") as f:
            total_lines = sum(1 for _ in f)

    # ===== чтение =====
    with in_path.open("rb") as f:
        for raw in tqdm(f, total=total_lines, unit="lines", desc="Reading"):
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                bad += 1
                continue

            emb = rec.get("embedding")
            if emb is None:
                continue

            try:
                src = int(rec["source_id"])
                pid = int(rec["id"])
            except Exception:
                bad += 1
                continue

            try:
                v = np.asarray(emb, dtype=np.float32)
            except Exception:
                bad += 1
                continue

            if dim is None:
                dim = int(v.size)
            elif v.size != dim:
                bad += 1
                continue

            vecs_map[src].append(v)
            ids_map[src].append(pid)

            if need_years:
                years_map[src].append(rec.get("date_year"))

    if dim is None:
        print("[ERR] Во входном нет валидных эмбеддингов.", file=sys.stderr)
        sys.exit(2)

    print(f"[OK] sources: {len(vecs_map)}; dim={dim}; плохих строк: {bad}")

    # ===== подготовка нормированных матриц по source =====
    Xnorm = {}
    for src, lst in tqdm(vecs_map.items(), desc="Normalize", unit="src"):
        X = np.vstack(lst).astype(np.float32, copy=False)
        Xhat = np.vstack([l2norm(x) for x in X])
        Xnorm[src] = Xhat

    # ===== запись по каждому режиму =====
    for name, params in mode_specs:
        out_path = out_dir / f"{base}_{name}.jsonl"
        wrote = 0
        with out_path.open("wb") as fout:
            it = tqdm(vecs_map.keys(), total=len(vecs_map), desc=f"Writing {name}", unit="src")
            for src in it:
                X = Xnorm[src]
                all_ids = ids_map[src]

                rec_out = {
                    "source_id": int(src),
                    "ids": [int(i) for i in all_ids],
                    "method": f"{name}" + (f"@{','.join(params)}" if params else "")
                }

                if name == "kmeans":
                    k = int(round(float(params[0]))) if params else 4
                    labels, centers = spherical_kmeans(X, k=k, iters=30, seed=args.seed)
                    sizes = np.bincount(labels, minlength=centers.shape[0]).astype(int).tolist()
                    weights = (np.array(sizes, dtype=np.float32) / float(len(all_ids))).tolist()
                    rec_out["embeddings"] = [c.tolist() for c in centers]
                    rec_out["weights"] = weights
                    rec_out["cluster_sizes"] = sizes

                elif name == "cmedoids":
                    k = int(round(float(params[0]))) if params else 4
                    labels, centers = spherical_kmeans(X, k=k, iters=30, seed=args.seed)
                    protos, proto_ids = [], []
                    for j in range(centers.shape[0]):
                        mask = (labels == j)
                        if not np.any(mask):
                            continue
                        Xj = X[mask]
                        ids_j = np.array(all_ids, dtype=np.int64)[mask]
                        sims = Xj @ centers[j]
                        i = int(np.argmax(sims))
                        protos.append(Xj[i].tolist())
                        proto_ids.append(int(ids_j[i]))
                    rec_out["embeddings"] = protos
                    rec_out["proto_ids"]  = proto_ids
                    rec_out["cluster_sizes"] = np.bincount(labels, minlength=centers.shape[0]).astype(int).tolist()

                elif name == "fps":
                    m = int(round(float(params[0]))) if params else 4
                    sel = furthest_point_sampling(X, m=m, seed=args.seed)
                    rec_out["embeddings"] = [X[i].tolist() for i in sel]
                    rec_out["proto_ids"]  = [int(all_ids[i]) for i in sel]

                elif name == "softk":
                    # softk@k,tau,iters
                    k   = int(round(float(params[0]))) if len(params) >= 1 else 4
                    tau = float(params[1]) if len(params) >= 2 else 10.0
                    it  = int(round(float(params[2]))) if len(params) >= 3 else 10
                    centers, weights = soft_kmeans(X, k=k, tau=tau, iters=it, seed=args.seed)
                    rec_out["embeddings"] = [c.tolist() for c in centers]
                    rec_out["weights"]    = weights.tolist()

                elif name == "pca":
                    r = int(round(float(params[0]))) if params else 3
                    comps, Sv = pca_components(X, r=r)
                    rec_out["embeddings"] = [c.tolist() for c in comps]
                    rec_out["weights"]    = Sv.tolist()

                elif name == "byyear":
                    minN = int(round(float(params[0]))) if params else 5
                    yrs = years_map[src]
                    if yrs is None:
                        # если в данных нет date_year — пропускаем
                        rec_out["embeddings"] = []
                        rec_out["year_keys"] = []
                        rec_out["year_counts"] = []
                    else:
                        Y = np.array(yrs)
                        uniq = [int(y) for y in sorted(set([yy for yy in Y if yy is not None]))]
                        protos, ykeys, ycnts = [], [], []
                        for y in uniq:
                            mask = (Y == y)
                            cnt = int(mask.sum())
                            if cnt >= minN:
                                proto = l2norm(X[mask].mean(axis=0))
                                protos.append(proto.tolist())
                                ykeys.append(int(y))
                                ycnts.append(cnt)
                        rec_out["embeddings"] = protos
                        rec_out["year_keys"] = ykeys
                        rec_out["year_counts"] = ycnts

                else:
                    # неизвестный режим — пропускаем
                    continue

                fout.write(orjson.dumps(rec_out) + b"\n")
                wrote += 1

        print(f"✔ {name}: {wrote} записей → {out_path}")


if __name__ == "__main__":
    main()
