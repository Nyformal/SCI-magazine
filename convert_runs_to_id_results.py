#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_runs_to_id_results.py

Ищет *.jsonl в указанных папках и конвертирует строки формата:
  {"model_file": "...", "model":"...", "query_id":232, "year_bucket":2019,
   "top":[{"rank":1,"source_id":57705,"score":...}, ...]}
в формат:
  {"id": 232, "results": [57705, ...]}

Опции:
  --recursive       рекурсивный обход директорий
  --inplace         перезаписывать исходные файлы (иначе создавать *.converted.jsonl)
  --glob PATTERN    маска файлов, по умолчанию *.jsonl
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Dict

# быстрый json + fallback
try:
    import orjson as _json
    def dumps(obj) -> bytes: return _json.dumps(obj)
    def loads(b: bytes): return _json.loads(b)
except Exception:
    import json as _json
    def dumps(obj) -> bytes: return _json.dumps(obj, ensure_ascii=False).encode("utf-8")
    def loads(b: bytes): return _json.loads(b.decode("utf-8"))

import tempfile
import os

def iter_files(roots: List[Path], pattern: str, recursive: bool) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            if root.match(pattern) or pattern == "*":
                yield root
            continue
        if recursive:
            yield from root.rglob(pattern)
        else:
            yield from root.glob(pattern)

def convert_line(raw: bytes) -> bytes | None:
    """
    Конвертирует одну строку JSONL.
    Возвращает байты новой строки (с финальным \n) или None — если строка пустая/битая.
    """
    s = raw.strip()
    if not s:
        return None
    try:
        obj = loads(s)
    except Exception:
        return None

    # ожидаем поля query_id и top (список объектов с source_id)
    try:
        qid = int(obj["query_id"])
        top = obj.get("top") or []
        # извлекаем source_id в ТОМ ЖЕ ПОРЯДКЕ, как они идут в массиве
        res_ids: List[int] = []
        seen = set()
        for item in top:
            sid = int(item["source_id"])
            # если важно избегать дублей — раскомментируй следующие 2 строки
            if sid in seen:
                continue
            seen.add(sid)
            res_ids.append(sid)
    except Exception:
        return None

    out = {"id": qid, "results": res_ids}
    return dumps(out) + b"\n"

def convert_file(path: Path, inplace: bool) -> None:
    """
    Конвертирует один .jsonl файл построчно.
    Если inplace=False — пишет рядом path.with_suffix('.converted.jsonl')
    Если inplace=True — пишет во временный файл и атомарно заменяет исходный.
    """
    if inplace:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        out_path = tmp_path
    else:
        out_path = path.with_suffix(path.suffix + ".converted.jsonl")

    total = 0
    ok = 0
    with open(path, "rb") as fin, open(out_path, "wb") as fout:
        for line in fin:
            total += 1
            converted = convert_line(line)
            if converted is not None:
                fout.write(converted)
                ok += 1

    if inplace:
        # атомарная замена
        os.replace(str(out_path), str(path))
        print(f"[OK] {path}  ({ok}/{total} строк сконвертировано)")
    else:
        print(f"[OK] {path} -> {out_path}  ({ok}/{total} строк сконвертировано)")

def main():
    ap = argparse.ArgumentParser(description="Конвертер JSONL в формат {id, results}")
    ap.add_argument("paths", nargs="+", help="Папки или файлы для обработки")
    ap.add_argument("--recursive", action="store_true", help="Рекурсивный обход директорий")
    ap.add_argument("--inplace", action="store_true", help="Перезаписать исходные файлы")
    ap.add_argument("--glob", default="*.jsonl", help="Маска файлов (по умолчанию *.jsonl)")
    args = ap.parse_args()

    roots = [Path(p).resolve() for p in args.paths]
    files = sorted(set(p for p in iter_files(roots, args.glob, args.recursive) if p.is_file()))

    if not files:
        print("[WARN] Файлов не найдено.", file=sys.stderr)
        return

    for f in files:
        try:
            convert_file(f, inplace=args.inplace)
        except Exception as e:
            print(f"[ERR] {f}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
