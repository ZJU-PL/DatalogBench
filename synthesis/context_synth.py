import argparse
import json

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "benchmark"
DATA_DIR = BENCHMARK_DIR / "data"


@dataclass
class SynthesisTask:
    case_id: str
    nl_query: str
    schema_def: Dict[str, List[Dict[str, str]]]
    # Optional auxiliary field stating what the task's terms and values mean.
    # Carried here rather than looked up at prompt time so the task context
    # stays the single description of what a solver is shown.
    knowledge: str = ""


def _parse_signature(signature: str) -> Dict[str, List[str]]:
    relation_name = signature.split("(", 1)[0].strip()
    inner = signature.split("(", 1)[1].rsplit(")", 1)[0]
    args = [x.strip() for x in inner.split(",") if x.strip()]
    return {"relation": relation_name, "args": args}


def _load_dataset() -> List[Dict]:
    dataset_file = BENCHMARK_DIR / "dataset.json"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with dataset_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _filter_records(records: List[Dict], dataset: str) -> List[Dict]:
    if not dataset or dataset.lower() == "all":
        return records

    filtered = [
        x for x in records
        if x.get("category") == dataset or x.get("sub_category") == dataset
    ]
    return filtered


def build_task(dataset: str, case_id: str) -> SynthesisTask:

    records = _filter_records(_load_dataset(), dataset)
    record = next((x for x in records if x.get("id") == case_id), None)
    if record is None:
        raise ValueError(f"Case id not found under dataset filter '{dataset}': {case_id}")

    schema_def = {
        "input": [
            {
                "signature": sig,
                "relation": _parse_signature(sig)["relation"],
                "args": _parse_signature(sig)["args"],
                "description": desc,
            }
            for sig, desc in record.get("input_relation", {}).items()
        ],
        "output": [
            {
                "signature": sig,
                "relation": _parse_signature(sig)["relation"],
                "args": _parse_signature(sig)["args"],
                "description": desc,
            }
            for sig, desc in record.get("output_relation", {}).items()
        ],
    }

    return SynthesisTask(
        case_id=case_id,
        nl_query=record.get("question", ""),
        schema_def=schema_def,
        knowledge=(record.get("knowledge") or "").strip(),
    )


def build_tasks(dataset: str = "all", case_id: Optional[str] = None) -> List[SynthesisTask]:
    records = _filter_records(_load_dataset(), dataset)
    ids = [x.get("id") for x in records if x.get("id")]
    if case_id:
        ids = [x for x in ids if x == case_id]
    if not ids:
        raise ValueError(f"No cases matched dataset filter '{dataset}'")
    tasks = [build_task(dataset=dataset, case_id=x) for x in ids]
    return tasks


def task_to_dict(task: SynthesisTask) -> Dict:
    return asdict(task)


def write_tasks_json(dataset: str = "all", case_id: Optional[str] = None) -> Path:
    tasks = build_tasks(dataset=dataset, case_id=case_id)
    payload = [task_to_dict(task) for task in tasks]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{case_id}.json" if case_id else f"{dataset}.json"
    out_path = DATA_DIR / file_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build normalized SynthesisTask objects from benchmark dataset metadata.")
    parser.add_argument("--dataset", type=str, default="all", help="Case filter value from category, sub category, or all")
    parser.add_argument("--case_id", type=str, default=None, help="Optional case id to build a single task")
    args = parser.parse_args()

    out_path = write_tasks_json(dataset=args.dataset, case_id=args.case_id)
    print(f"[DONE] wrote tasks -> {out_path}")


if __name__ == "__main__":
    main()
