# pip install psycopg2-binary
import json
from pathlib import Path
import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "dbname":   "scopusdb",
    "user":     "postgres",
    "password": "bartbartsimpson123",
    "host":     "127.0.0.1",
    "port":     "5432",
}

TABLE_NAME = "table_3_2"
OUT_DIR = Path("export_jsonl")
BATCH_SIZE = 10_000  # размер пачки для стриминга

def normalize_source_type(s: str | None) -> str | None:
    if s is None:
        return None
    # нормализация: без лишних пробелов, lower, заменяем '&' на 'and'
    t = " ".join(s.lower().replace("&", "and").split())
    if t == "journal":
        return "journal"
    if t in {"conference and proceedings", "conference proceedings"}:
        return "conference and proceedings"
    return None  # другие типы нам не нужны

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_journal = OUT_DIR / "table_3_2_journal.jsonl"
    path_confproc = OUT_DIR / "table_3_2_conference_and_proceedings.jsonl"

    with psycopg2.connect(**DB_CONFIG) as conn, \
         conn.cursor(name="cur_stream", cursor_factory=psycopg2.extras.DictCursor) as cur, \
         open(path_journal, "w", encoding="utf-8") as f_journal, \
         open(path_confproc, "w", encoding="utf-8") as f_conf:

        # Серверный курсор для стриминга — не грузим всю таблицу в память
        cur.itersize = BATCH_SIZE
        cur.execute(f"""
            SELECT id, publication_source_id, date_year, source_type
            FROM {TABLE_NAME}
        """)

        cnt_total = cnt_journal = cnt_conf = cnt_skipped = 0

        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break

            for r in rows:
                cnt_total += 1
                stype = normalize_source_type(r["source_type"])
                if stype not in {"journal", "conference and proceedings"}:
                    cnt_skipped += 1
                    continue

                obj = {
                    "id": int(r["id"]),
                    "source_id": int(r["publication_source_id"]),
                    "date_year": None if r["date_year"] is None else int(r["date_year"]),
                }
                if stype == "journal":
                    f_journal.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    cnt_journal += 1
                else:
                    f_conf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    cnt_conf += 1

    print("Готово.")
    print(f"Всего строк прочитано: {cnt_total}")
    print(f"В journal: {cnt_journal} → {path_journal}")
    print(f"В conference and proceedings: {cnt_conf} → {path_confproc}")
    print(f"Пропущено (другие source_type/NULL): {cnt_skipped}")

if __name__ == "__main__":
    main()
