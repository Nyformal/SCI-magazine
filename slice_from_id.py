# slice_from_id.py
# Пример запуска:
#   python slice_from_id.py --input journal_embeddings.jsonl --id 1558159 --output journal_from_1558159.jsonl
#
# Опции:
#   --skip-first   если указать — начнёт со СЛЕДУЮЩЕЙ строки после найденной (на случай,
#                  если первая с этим id обрезана и её не надо включать)

import argparse
import json
import os
from pathlib import Path
import sys

def slice_from_id(input_path: Path, output_path: Path, start_id: int, skip_first: bool = False) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.exists():
        print(f"[ERROR] Файл не найден: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Чтобы запись была атомарной
    tmp_out = output_path.with_suffix(output_path.suffix + ".tmp")

    found = False
    found_line_no = None
    first_written = 0

    id_patterns = (f'"id": {start_id}', f'"id":{start_id}')  # быстрый фильтр по подстроке

    with input_path.open("rt", encoding="utf-8", errors="strict") as fin, \
         tmp_out.open("wt", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            if not found:
                # Быстрая проверка по подстроке
                if not any(p in line for p in id_patterns):
                    continue

                # Если подстрока есть — пытаемся подтвердить JSON’ом
                is_valid_json = False
                try:
                    obj = json.loads(line)
                    is_valid_json = True
                except Exception:
                    obj = None

                if (is_valid_json and obj.get("id") == start_id) or (not is_valid_json):
                    found = True
                    found_line_no = line_no

                    # если нужно пропустить первую (например, она обрезана)
                    if skip_first:
                        continue

                    # включаем найденную строку как есть (даже если она обрезана)
                    fout.write(line)
                    first_written = 1
            else:
                # уже нашли — просто копируем остальное как есть
                fout.write(line)

    if not found:
        print(f"[ERROR] В файле {input_path.name} не найден id={start_id}", file=sys.stderr)
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        sys.exit(2)

    # Финализируем файл
    os.replace(tmp_out, output_path)

    print(f"[OK] Найден id={start_id} на строке {found_line_no}.")
    if first_written == 1:
        print(f"[OK] Первая записанная строка — та же, где найден id (включена).")
    else:
        print(f"[OK] Первая записанная строка — следующая после найденной (из-за --skip-first).")
    print(f"[OK] Итоговый файл: {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Сделать новый JSONL, начиная со строки, где встречается указанный id, и далее до конца файла.")
    ap.add_argument("--input", required=True, help="входной .jsonl (например, journal_embeddings.jsonl)")
    ap.add_argument("--output", required=True, help="выходной .jsonl (например, journal_from_1558159.jsonl)")
    ap.add_argument("--id", type=int, required=True, help="id публикации, с которой начинать")
    ap.add_argument("--skip-first", action="store_true", help="начать со следующей строки (не включать найденную)")
    args = ap.parse_args()

    slice_from_id(Path(args.input), Path(args.output), args.id, skip_first=args.skip_first)

if __name__ == "__main__":
    main()
