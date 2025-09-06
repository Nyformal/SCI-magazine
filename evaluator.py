from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, Iterable, Tuple, List
import pandas as pd

try:
    from tqdm import tqdm
    def pbar(iterable, **kw): return tqdm(iterable, **kw)
except Exception:
    def pbar(iterable, **kw): return iterable  # fallback, без прогресса

KS_DEFAULT = (1, 3, 5, 10, 20)

# ----------------- utils -----------------

def find_first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def load_truth_map(truth_path: Path) -> Dict[int, int]:
    """
    Быстро загружает id -> source_id из JSONL.
    """
    mapping: Dict[int, int] = {}
    with truth_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get('id')
            sid = rec.get('source_id')
            if pid is not None and sid is not None:
                mapping[int(pid)] = int(sid)
    return mapping

def iter_pred_files(models_dir: Path) -> Iterable[Tuple[str, Path]]:
    tasks = [
        ("journal",   models_dir / "journal"   / "v1_results"),
        ("journal",   models_dir / "journal"   / "v2_results"),
        ("conference", models_dir / "conference" / "v1_results"),
        ("conference", models_dir / "conference" / "v2_results"),
    ]
    seen = set()
    for task, root in tasks:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            key = (task, path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield task, path

# ----------------- metrics (потоково) -----------------

class MetricsAgg:
    """
    Потоковый аккумулятор метрик для одной модели/файла.
    """
    def __init__(self, ks: Tuple[int, ...] = KS_DEFAULT):
        self.ks = tuple(sorted(set(ks)))
        self.n = 0  # число записей, по которым была истина
        self.mrr_at_5 = 0.0
        self.mrr_at_10 = 0.0
        self.hr_counts = {k: 0 for k in self.ks}

    def update(self, true_id: int, neighbors: List[int]):
        self.n += 1

        # MRR@5
        for rank, vid in enumerate(neighbors[:5], start=1):
            if vid == true_id:
                self.mrr_at_5 += 1.0 / rank
                break

        # MRR@10
        for rank, vid in enumerate(neighbors[:10], start=1):
            if vid == true_id:
                self.mrr_at_10 += 1.0 / rank
                break

        # HR@K
        for k in self.ks:
            if true_id in neighbors[:k]:
                self.hr_counts[k] += 1

    def to_row(self, model_name: str) -> Dict[str, float | str]:
        row = {
            "model": model_name,
            "MRR@5":  self.mrr_at_5 / self.n if self.n else 0.0,
            "MRR@10": self.mrr_at_10 / self.n if self.n else 0.0,
        }
        # абсолюты
        for k in self.ks:
            row[f"HR@{k}"] = self.hr_counts[k]
        # проценты
        for k in self.ks:
            row[f"HR@{k}_%"] = (self.hr_counts[k] / self.n) if self.n else 0.0
        # служебное
        row["N_evaluated"] = self.n
        return row

# ----------------- core -----------------

def eval_file(pred_path: Path, truth_map: Dict[int, int], ks: Tuple[int, ...]) -> Dict[str, float | str]:
    """
    Считает метрики по одному файлу с предсказаниями.
    Файл формата: {"id": 232, "results": [57705, ...]}
    """
    agg = MetricsAgg(ks)
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("id")
            if pid is None:
                continue
            pid = int(pid)
            true_id = truth_map.get(pid)
            if true_id is None:
                continue  # нет истины — пропускаем
            neigh = rec.get("results") or []
            # приводим к int, режем по максимуму нужному (для ускорения)
            # максимальный k из набора и 10 для MRR@10
            need = max(max(ks), 10)
            neighbors: List[int] = []
            for x in neigh[:need]:
                try:
                    neighbors.append(int(x))
                except Exception:
                    continue
            agg.update(true_id=int(true_id), neighbors=neighbors)
    model_name = str(pred_path.as_posix())
    return agg.to_row(model_name=model_name)

def main():
    ap = argparse.ArgumentParser(description="Оценка MRR/HR для файлов предсказаний и выгрузка в XLSX")
    ap.add_argument("--models-dir", default="models", help="Корневая папка с моделями (по умолчанию: models)")
    ap.add_argument("--data-dir", default="data", help="Папка с истиной (по умолчанию: data)")
    ap.add_argument("--truth-conference", default=None, help="Путь к conference.jsonl (если не указан — ищем автоматически)")
    ap.add_argument("--truth-journal", default=None, help="Путь к journal.jsonl (если не указан — ищем автоматически)")
    ap.add_argument("--out", default="metrics.xlsx", help="Имя выходного файла .xlsx")
    ap.add_argument("--ks", default="1,3,5,10,20", help="Список K через запятую для HR@K (и макс.K влияет на срез results)")
    args = ap.parse_args()

    ks = tuple(sorted({int(x) for x in args.ks.split(",") if x.strip()}))
    models_dir = Path(args.models_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    # --- находим truth файлы ---
    conf_path = Path(args.truth_conference) if args.truth_conference else find_first_existing([
        data_dir / "conference.jsonl",
        data_dir / "pred" / "conference.jsonl",
    ])
    journ_path = Path(args.truth_journal) if args.truth_journal else find_first_existing([
        data_dir / "journal.jsonl",
        data_dir / "pred" / "journal.jsonl",
    ])
    if conf_path is None or not conf_path.exists():
        raise SystemExit("Не найден conference.jsonl (ни data/conference.jsonl, ни data/pred/conference.jsonl). Укажите --truth-conference")
    if journ_path is None or not journ_path.exists():
        raise SystemExit("Не найден journal.jsonl (ни data/journal.jsonl, ни data/pred/journal.jsonl). Укажите --truth-journal")

    print(f"[*] Loading truth: {conf_path}")
    truth_conference = load_truth_map(conf_path)
    print(f"    conference truths: {len(truth_conference):,}")

    print(f"[*] Loading truth: {journ_path}")
    truth_journal = load_truth_map(journ_path)
    print(f"    journal truths: {len(truth_journal):,}")

    rows: List[Dict[str, float | str]] = []

    # --- обходим все файлы предсказаний ---
    for task, pred_path in pbar(list(iter_pred_files(models_dir)), desc="Evaluating", unit="file"):
        truth_map = truth_conference if task == "conference" else truth_journal
        row = eval_file(pred_path, truth_map, ks)
        # Более читаемое имя модели: относительный путь от models_dir
        row["model"] = f"{task}/{pred_path.relative_to(models_dir).as_posix()}"
        rows.append(row)

    if not rows:
        raise SystemExit("Не найдено файлов предсказаний (*.jsonl) под models/...")

    # --- собираем общий DataFrame с нужным порядком колонок ---
    cols = ["model", "MRR@5", "MRR@10"]
    cols += [f"HR@{k}" for k in ks]
    cols += [f"HR@{k}_%" for k in ks]
    cols += ["N_evaluated"]

    df = pd.DataFrame(rows)
    # добавим отсутствующие столбцы, если какие-то ks не встретились
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[cols]

    # --- сохраняем в xlsx ---
    out_path = Path(args.out)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # первая строка — это заголовки DataFrame (model + метрики)
        df.to_excel(writer, index=False, sheet_name="Metrics")

    print(f"[OK] Saved → {out_path.resolve()}")

if __name__ == "__main__":
    main()
