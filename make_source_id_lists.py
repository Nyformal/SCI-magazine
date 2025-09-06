import argparse
from pathlib import Path
from collections import defaultdict

try:
    import orjson
    def jloads(b: bytes): return orjson.loads(b)
    def jdumps(obj) -> bytes: return orjson.dumps(obj)
except ImportError:
    import json
    def jloads(b: bytes):
        return json.loads(b.decode("utf-8"))
    def jdumps(obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

def process_file(in_path: Path) -> Path:
    if not in_path.exists():
        print(f"файл не найден: {in_path}")
        return None

    groups = defaultdict(list)  # source_id -> [ids]
    total_lines = 0
    bad_lines = 0

    with in_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            total_lines += 1
            try:
                rec = jloads(raw)
            except Exception:
                bad_lines += 1
                continue
            sid = rec.get("source_id")
            pid = rec.get("id")
            if sid is None or pid is None:
                bad_lines += 1
                continue
            try:
                sid = int(sid)
                pid = int(pid)
            except Exception:
                bad_lines += 1
                continue
            groups[sid].append(pid)

    # уникализируем и сортируем ids
    for sid, lst in list(groups.items()):
        uniq = sorted(set(lst))
        groups[sid] = uniq

    # сводка
    total_sources = len(groups)
    ge_5 = sum(1 for v in groups.values() if len(v) >= 5)
    ge_10 = sum(1 for v in groups.values() if len(v) >= 10)

    # запись
    out_path = in_path.with_name(in_path.stem + "_source_ids.jsonl")
    with out_path.open("wb") as fout:
        for sid, ids in groups.items():
            obj = {"source_id": sid, "ids": ids}
            fout.write(jdumps(obj) + b"\n")

    print(f"{in_path.name}: строк={total_lines}, битых={bad_lines}")
    print(f"   source_id: всего={total_sources}, ≥5={ge_5}, ≥10={ge_10}")
    print(f"   → записано: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(
        description="Собрать списки публикаций по source_id из *.jsonl"
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["journal.jsonl", "conference.jsonl"],
        help="Список входных JSONL (по умолчанию journal.jsonl conference.jsonl в директории скрипта)."
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    for name in args.inputs:
        in_path = (base / name).resolve() if not Path(name).is_absolute() else Path(name)
        process_file(in_path)

if __name__ == "__main__":
    main()
