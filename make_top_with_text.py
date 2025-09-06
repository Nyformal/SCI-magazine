#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Iterable, Optional
import gzip
import orjson
from tqdm import tqdm
import sys
import fnmatch

PLACEHOLDER = "[краткое описание не найдено]"

# ---------- helpers ----------

def open_rb(path: Path):
    """Открыть файл в бинарном режиме, поддержка .gz."""
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")

def try_candidates(name: str, base_dirs: List[Path]) -> Optional[Path]:
    """
    Ищет файл по ряду эвристик:
      - ровно name
      - name + .jsonl
      - name + .jsonl.gz
      - шаблон name*.jsonl* в base_dirs
    Возвращает первый найденный абсолютный Path или None.
    """
    # 1) ровно как передано (абсолютный/относительный)
    p = Path(name)
    if p.exists():
        return p.resolve()

    # 2) добавить расширения
    for ext in (".jsonl", ".jsonl.gz"):
        q = Path(name + ext)
        if q.exists():
            return q.resolve()

    # 3) поиск по шаблону в заданных папках
    patterns = [f"{name}*.jsonl", f"{name}*.jsonl.gz"]
    for d in base_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            for cand in d.glob(pat):
                if cand.is_file():
                    return cand.resolve()
    return None

def resolve_unique_path(user_arg: str) -> Path:
    """
    Пытается найти unique_text по имени из --unique.
    Ищет: как есть, с .jsonl/.jsonl.gz, и по шаблонам в:
      - текущей папке (cwd)
      - папке скрипта (__file__)
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    p = try_candidates(user_arg, [cwd, script_dir])
    if p is None:
        # последняя попытка: если user_arg — чистое имя без путей, ищем в cwd и script_dir любой *.jsonl*
        base = Path(user_arg).name
        p = try_candidates(base, [cwd, script_dir])
    if p is None:
        raise FileNotFoundError(f"Не найден файл текстов по шаблону: {user_arg} "
                                f"(пробовал как есть, с .jsonl/.jsonl.gz и {user_arg}*.jsonl*)")
    return p

def load_explanations(path: Path, count_lines: bool = False) -> Dict[int, str]:
    """Собираем словарь id -> explanation из unique_text(.jsonl|.jsonl.gz)."""
    text_by_id: Dict[int, str] = {}

    # точный total только для не-сжатых
    total = None
    if count_lines and path.suffix != ".gz":
        with path.open("rb") as f:
            total = sum(1 for _ in f)

    with open_rb(path) as f:
        it = tqdm(f, total=total, unit="lines", desc=f"Load texts {path.name}") if count_lines else f
        for raw in it:
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                continue
            _id = rec.get("id")
            if _id is None:
                continue
            expl = rec.get("explanation", "")
            if not isinstance(expl, str) or expl.strip() == PLACEHOLDER:
                expl = ""
            try:
                text_by_id[int(_id)] = expl
            except Exception:
                continue
    return text_by_id

def transform_file(input_path: Path, text_by_id: Dict[int, str], outdir: Path | None, count_lines: bool = False) -> Path:
    """
    Преобразует файл вида:
      {"id": 232, "results": [2603759, 1002068, ...]}
    в строки:
      {"query_id": "232", "explanation": "...", "results": [{"id": 2603759, "explanation": "..."}, ...]}
    """
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{input_path.stem}_with_text.jsonl"
    else:
        out_path = input_path.with_name(f"{input_path.stem}_with_text.jsonl")

    total = None
    if count_lines and input_path.suffix != ".gz":
        with input_path.open("rb") as f:
            total = sum(1 for _ in f)

    written = 0
    with input_path.open("rb") as fin, out_path.open("wb") as fout:
        it = tqdm(fin, total=total, unit="lines", desc=f"Transform {input_path.name}") if count_lines else fin
        for raw in it:
            if not raw.strip():
                continue
            try:
                rec = orjson.loads(raw)
            except Exception:
                continue

            qid = rec.get("id")
            results = rec.get("results")
            if qid is None or not isinstance(results, list):
                continue

            try:
                qid_int = int(qid)
            except Exception:
                qid_int = None

            q_text = text_by_id.get(qid_int, "") if qid_int is not None else ""

            out_results: List[dict] = []
            for rid in results:
                try:
                    rid_int = int(rid)
                except Exception:
                    continue
                r_text = text_by_id.get(rid_int, "")
                out_results.append({"id": rid_int, "explanation": r_text})

            out_obj = {
                "query_id": str(qid) if qid is not None else "",
                "explanation": q_text,
                "results": out_results,
            }
            fout.write(orjson.dumps(out_obj) + b"\n")
            written += 1

    print(f"✔ {input_path.name}: {written} строк → {out_path}")
    return out_path

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Склеить тексты из unique_text(.jsonl|.jsonl.gz) к topK результатам."
    )
    ap.add_argument("--unique", type=str, required=True,
                    help="Имя/путь к unique_text (можно без расширения, поддержка .jsonl и .jsonl.gz).")
    ap.add_argument("--inputs", type=str, nargs="+", required=True,
                    help="Входные *_test_top1000.jsonl файлы (можно несколько).")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Папка для вывода. По умолчанию — рядом с каждым входным.")
    ap.add_argument("--count-lines", action="store_true",
                    help="Точные tqdm (для .gz отключится автоматически).")
    return ap.parse_args()

def main():
    args = parse_args()

    try:
        unique_path = resolve_unique_path(args.unique)
    except FileNotFoundError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)

    inputs = [Path(p).resolve() for p in args.inputs]
    outdir = Path(args.outdir).resolve() if args.outdir else None

    # 1) грузим карту id->explanation
    text_by_id = load_explanations(unique_path, count_lines=args.count_lines)
    print(f"[*] loaded {len(text_by_id):,} texts from {unique_path}")

    # 2) прогоняем все входы
    for ip in inputs:
        transform_file(ip, text_by_id, outdir=outdir, count_lines=args.count_lines)

if __name__ == "__main__":
    main()
