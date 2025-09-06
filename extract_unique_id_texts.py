#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_unique_id_texts.py

1) Собирает все уникальные ID из JSONL файлов формата:
   {"id": 232, "results": [2603759, 1002068, ...]}

2) Делает батч-запросы к Postgres (public.publication) и пишет JSONL:
   {"id": 232, "explanation": "..."}

Особенности:
- explanation = abstract, если пусто → name
- Пустые тексты не пишутся вовсе (фильтрация в SQL)
- Работает чанками, не раздувает память

Зависимости: psycopg2-binary, tqdm, (опц.) orjson
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Быстрый JSON с fallback
try:
    import orjson as _json
    def dumps(obj) -> bytes: return _json.dumps(obj)
    def loads(b: bytes): return _json.loads(b)
except Exception:
    import json as _json
    def dumps(obj) -> bytes: return _json.dumps(obj, ensure_ascii=False).encode("utf-8")
    def loads(b: bytes): return _json.loads(b.decode("utf-8"))

import psycopg2
import psycopg2.extras
from tqdm import tqdm

DB_CONFIG = {
    "dbname":   "scopusdb",
    "user":     "postgres",
    "password": "bartbartsimpson123",
    "host":     "127.0.0.1",
    "port":     "5432",
}

# SQL: берём abstract, если он пустой — name; отбрасываем пустые вообще
SQL_FETCH = """
SELECT id,
       COALESCE(NULLIF(BTRIM(abstract), ''), NULLIF(BTRIM(name), '')) AS text
FROM public.publication
WHERE id = ANY(%s)
  AND COALESCE(NULLIF(BTRIM(abstract), ''), NULLIF(BTRIM(name), '')) IS NOT NULL
  AND COALESCE(NULLIF(BTRIM(abstract), ''), NULLIF(BTRIM(name), '')) <> ''
"""

def connect():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True  # не держим транзакции, читаем быстрее
    return conn

def parse_input_ids(paths: List[Path]) -> List[int]:
    """Парсит входные JSONL и возвращает отсортированный список уникальных ID (int)."""
    uniq = set()
    for p in paths:
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = loads(line)
                except Exception:
                    continue
                # query id
                try:
                    qid = int(obj["id"])
                    uniq.add(qid)
                except Exception:
                    pass
                # results
                res = obj.get("results")
                if isinstance(res, list):
                    for rid in res:
                        try:
                            uniq.add(int(rid))
                        except Exception:
                            continue
    return sorted(uniq)

def fetch_texts_chunk(conn, ids: List[int]) -> Dict[int, str]:
    """Возвращает словарь id -> текст для данного чанка ID (уже без пустых)."""
    out: Dict[int, str] = {}
    if not ids:
        return out
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(SQL_FETCH, (ids,))
        for row in cur:
            _id = int(row["id"])
            txt = row["text"] if row["text"] is not None else ""
            # SQL уже фильтрует пустые, но на всякий — лишний чекап:
            if txt:
                out[_id] = txt
    return out

def main():
    ap = argparse.ArgumentParser(description="Выгрузка explanation по уникальным ID из БД (abstract → name)")
    ap.add_argument("inputs", nargs="+",
                    help="Входные JSONL (например, conference_test_top1000.jsonl journal_test_top1000.jsonl)")
    ap.add_argument("--out", default="unique_texts.jsonl",
                    help="Имя выходного JSONL (по умолчанию unique_texts.jsonl)")
    ap.add_argument("--batch", type=int, default=50000,
                    help="Размер чанка ID на один SQL-запрос (default: 50000)")
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    for p in in_paths:
        if not p.exists():
            print(f"[WARN] Нет файла: {p}", file=sys.stderr)

    print("[*] Собираю уникальные ID...")
    unique_ids = parse_input_ids(in_paths)
    total_ids = len(unique_ids)
    print(f"[*] Уникальных ID: {total_ids:,}")

    if total_ids == 0:
        print("[WARN] Уникальных ID не найдено. Завершение.")
        return

    conn = connect()
    try:
        with open(args.out, "wb") as fout:
            for i in tqdm(range(0, total_ids, args.batch), desc="Query & write", unit="batch"):
                chunk = unique_ids[i:i+args.batch]
                text_map = fetch_texts_chunk(conn, chunk)  # уже без пустых

                # Пишем только те ID, что реально нашлись и имеют непустой текст
                for _id, txt in text_map.items():
                    rec = {"id": int(_id), "explanation": txt}
                    fout.write(dumps(rec) + b"\n")
    finally:
        conn.close()

    print(f"[OK] Готово → {args.out}")

if __name__ == "__main__":
    main()
