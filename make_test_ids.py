import argparse
import random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# быстрая (если есть) и обычная JSON-загрузка/выгрузка
try:
    import orjson
    def jloads(b: bytes): return orjson.loads(b)
    def jdumps(obj) -> bytes: return orjson.dumps(obj)
except ImportError:
    import json
    def jloads(b: bytes): return json.loads(b.decode("utf-8"))
    def jdumps(obj) -> bytes: return json.dumps(obj, ensure_ascii=False).encode("utf-8")

def round_half_up(n: float) -> int:
    return int(Decimal(n).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def process_one(input_path: Path, rng: random.Random) -> Path:
    if not input_path.exists():
        print(f"файл не найден: {input_path}")
        return None

    total_sources = 0
    rule_1_sources = 0     # 5..10 → 1 id
    rule_10p_sources = 0   # >10 → 10%
    selected_ids = []

    with input_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = jloads(raw)
            src = rec.get("source_id")
            ids = rec.get("ids") or []
            if src is None or not isinstance(ids, list) or not ids:
                continue

            ids = list({int(x) for x in ids})   # на всякий случай уникализируем
            cnt = len(ids)
            total_sources += 1

            k = 0
            if 5 <= cnt <= 10:
                rule_1_sources += 1
                k = 1
            elif cnt > 10:
                rule_10p_sources += 1
                k = max(1, round_half_up(cnt * 0.10))

            if k > 0:
                chosen = rng.sample(ids, k) if k < cnt else ids
                selected_ids.extend(chosen)

    # итоговый файл
    if input_path.name.endswith("_source_ids.jsonl"):
        out_name = input_path.name.replace("_source_ids.jsonl", "_test_ids.json")
    else:
        out_name = input_path.stem + "_test_ids.json"
    out_path = input_path.with_name(out_name)

    with out_path.open("wb") as fout:
        fout.write(jdumps(selected_ids) + b"\n")

    print(f"{input_path.name}: sources={total_sources}, по правилу 1шт={rule_1_sources}, по правилу 10%={rule_10p_sources}, "
          f"выбрано id={len(selected_ids)}")
    print(f"   → записано: {out_path}")
    return out_path

def main():
    p = argparse.ArgumentParser(description="Собрать тестовые ID из *_source_ids.jsonl по правилам отбора")
    p.add_argument("--inputs", nargs="*", default=["journal_source_ids.jsonl", "conference_source_ids.jsonl"],
                   help="Список входных файлов *_source_ids.jsonl")
    p.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимой случайной выборки")
    args = p.parse_args()

    rng = random.Random(args.seed)
    for inp in args.inputs:
        path = Path(inp).resolve()
        process_one(path, rng)

if __name__ == "__main__":
    main()
