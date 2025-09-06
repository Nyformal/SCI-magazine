# split_by_year_ranges.py
# Создаёт 5 файлов 2019.jsonl, 2020.jsonl, 2021.jsonl, 2022.jsonl, 2023.jsonl
# в той же директории, где находится исходный файл.
#
# Пример:
#   python split_by_year_ranges.py --input journal_embeddings.jsonl
#
# Логика соответствия лет задана по матрице:
#   2019 -> {2019}
#   2020 -> {2019, 2020}
#   2021 -> {2019, 2020, 2021}
#   2022 -> {2019, 2020, 2021, 2022}
#   2023 -> {2020, 2021, 2022, 2023}
import argparse
import re
from pathlib import Path
from contextlib import ExitStack

YEAR_RANGES = {
    2019: {2019},
    2020: {2019, 2020},
    2021: {2019, 2020, 2021},
    2022: {2019, 2020, 2021, 2022},
    2023: {2020, 2021, 2022, 2023},
}

RE_YEAR = re.compile(r'"date_year"\s*:\s*"?(\d{4})"?')

def extract_year(line: str) -> int | None:
    m = RE_YEAR.search(line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Разложить JSONL по диапазонам лет в 5 файлов (2019..2023).")
    ap.add_argument("--input", required=True, help="путь к journal_embeddings.jsonl")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Не найден входной файл: {src}")

    out_dir = src.parent
    out_paths = {yr: out_dir / f"{yr}.jsonl" for yr in YEAR_RANGES.keys()}

    with ExitStack() as stack:
        writers = {yr: stack.enter_context(p.open("w", encoding="utf-8")) for yr, p in out_paths.items()}

        total = 0
        written = {yr: 0 for yr in YEAR_RANGES.keys()}

        # Читаем построчно и пишем исходную строку "как есть" (не пересериализуем JSON)
        with src.open("r", encoding="utf-8", errors="replace") as fin:
            for total, line in enumerate(fin, start=1):
                # пропустим пустые строки на всякий
                if not line.strip():
                    continue

                y = extract_year(line)
                if y is None:
                    # строки без date_year просто игнорируем
                    continue

                # Определяем, в какие файлы эта строка должна попасть
                for out_year, allowed in YEAR_RANGES.items():
                    if y in allowed:
                        writers[out_year].write(line)
                        written[out_year] += 1

                # Можно добавить простой прогресс раз в N строк (необязательно):
                # if total % 200000 == 0:
                #     print(f"processed {total} lines...", flush=True)

    print(f"[OK] Готово. Обработано строк: {total}")
    for yr in sorted(written):
        print(f"  {yr}.jsonl: {written[yr]} строк → {out_paths[yr]}")

if __name__ == "__main__":
    main()
