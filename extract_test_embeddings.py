import argparse
import os
import sys
from pathlib import Path

# быстрый JSON (если есть) или стандартный json
try:
    import orjson
    def jloads(b: bytes): return orjson.loads(b)
    def jdumps(obj) -> bytes: return orjson.dumps(obj)
except ImportError:
    import json
    def jloads(b: bytes): return json.loads(b.decode("utf-8"))
    def jdumps(obj) -> bytes: return json.dumps(obj, ensure_ascii=False).encode("utf-8")

def load_test_ids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл с тест-идшниками: {path}")
    data = jloads(path.read_bytes())
    if not isinstance(data, list):
        raise ValueError(f"{path} должен быть JSON-массивом id")
    ids = set()
    for x in data:
        try:
            ids.add(int(x))
        except Exception:
            pass
    return ids

def main():
    p = argparse.ArgumentParser(description="Вырезать test-строки из *_embeddings.jsonl и сложить их в *_test_embeddings.jsonl")
    p.add_argument("dataset", choices=["journal", "conference"], help="Какой датасет обрабатывать")
    p.add_argument("--dir", dest="dir", default=".", help="Каталог, где лежат файлы датасета (по умолчанию текущий)")
    p.add_argument("--no-backup", action="store_true", help="Не сохранять резервную копию исходного embeddings")
    p.add_argument("--dry-run", action="store_true", help="Только посчитать, ничего не менять")
    args = p.parse_args()

    base = Path(args.dir).resolve()
    name = args.dataset

    emb_path   = base / f"{name}_embeddings.jsonl"
    ids_path   = base / f"{name}_test_ids.json"
    out_test   = base / f"{name}_test_embeddings.jsonl"
    tmp_path   = base / f"{name}_embeddings.tmp.jsonl"
    backup_path= base / f"{name}_embeddings.backup.jsonl"

    if not emb_path.exists():
        print(f"❌ Не найден embeddings-файл: {emb_path}", file=sys.stderr)
        sys.exit(1)
    try:
        test_ids = load_test_ids(ids_path)
    except Exception as e:
        print(f"❌ Ошибка чтения {ids_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not test_ids:
        print("⚠️  В тестовом наборе нет id — делать нечего.", file=sys.stderr)
        sys.exit(0)

    print(f"▶ Обработка: {emb_path.name}")
    print(f"   Тестовых id: {len(test_ids)}")
    if args.dry_run:
        print("   Режим: dry-run (файлы не изменяются)")

    processed = 0
    moved = 0
    kept  = 0
    bad_lines = 0

    # откроем файлы (в dry-run только читаем)
    fout_test = None
    fout_tmp  = None
    try:
        with emb_path.open("rb") as fin:
            if not args.dry_run:
                fout_test = out_test.open("wb")
                fout_tmp  = tmp_path.open("wb")

            for raw in fin:
                if not raw.strip():
                    # пустые/пробельные — просто сохраняем в tmp, если не dry-run
                    processed += 1
                    if not args.dry_run:
                        fout_tmp.write(raw)
                    else:
                        kept += 1  # логически «остаётся»
                    continue

                try:
                    rec = jloads(raw)
                    rid = rec.get("id", None)
                    rid = int(rid) if rid is not None else None
                except Exception:
                    # битая строка — не рискуем, оставляем в основном файле
                    bad_lines += 1
                    processed += 1
                    if not args.dry_run:
                        fout_tmp.write(raw)
                    else:
                        kept += 1
                    continue

                processed += 1
                if rid in test_ids:
                    moved += 1
                    if not args.dry_run:
                        # пишем строку как есть в test_embeddings
                        fout_test.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    # не пишем её в tmp => будет удалена из embeddings
                else:
                    kept += 1
                    if not args.dry_run:
                        fout_tmp.write(raw)

        if args.dry_run:
            print(f"— Просмотрено строк: {processed}")
            print(f"— Нашлось тестовых: {moved}")
            print(f"— Останется в embeddings: {kept}")
            print(f"— Непарсибельных строк: {bad_lines}")
            return

        # безопасная замена исходника
        # сначала бэкап, если нужен
        if not args.no_backup:
            if backup_path.exists():
                backup_path.unlink()  # убираем старый бэкап
            os.replace(emb_path, backup_path)
        else:
            # удаляем оригинал сразу, чтобы можно было переименовать tmp поверх
            emb_path.unlink()

        # tmp -> embeddings
        os.replace(tmp_path, emb_path)

        print(f"✔ Готово.")
        print(f"— Всего строк: {processed}")
        print(f"— Перемещено в {out_test.name}: {moved}")
        print(f"— Осталось в {emb_path.name}: {kept}")
        print(f"— Непарсибельных строк: {bad_lines}")
        if not args.no_backup:
            print(f"— Резервная копия исходника: {backup_path}")

    finally:
        # аккуратно закрываем файлы, если открыты
        try:
            if fout_test: fout_test.close()
        except Exception:
            pass
        try:
            if fout_tmp: fout_tmp.close()
        except Exception:
            pass
        # если что-то пошло не так — удалим tmp
        if tmp_path.exists() and (args.dry_run or not emb_path.exists()):
            try:
                tmp_path.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    main()
