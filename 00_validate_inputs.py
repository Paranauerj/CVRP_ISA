"""Validate CVRP instances and paired BKS files without running optimization."""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DIMENSION_RE = re.compile(r"^DIMENSION\s*:\s*(\d+)\s*$", re.IGNORECASE)
CAPACITY_RE = re.compile(r"^CAPACITY\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
COST_RE = re.compile(r"^COST\s*:?\s*([-+0-9.eE]+)\s*$", re.IGNORECASE)


def read_header(path: Path) -> tuple[int, float]:
    dimension = None
    capacity = None
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            stripped = line.strip()
            dimension_match = DIMENSION_RE.match(stripped)
            capacity_match = CAPACITY_RE.match(stripped)
            if dimension_match:
                dimension = int(dimension_match.group(1))
            if capacity_match:
                capacity = float(capacity_match.group(1))
            if stripped == "NODE_COORD_SECTION":
                break
    if dimension is None or capacity is None:
        raise ValueError("missing DIMENSION or CAPACITY")
    if dimension < 2:
        raise ValueError("DIMENSION must include depot and at least one customer")
    return dimension - 1, capacity


def read_bks_cost(path: Path) -> float:
    cost = None
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            match = COST_RE.match(line.strip())
            if match:
                cost = float(match.group(1))
    if cost is None:
        raise ValueError("missing Cost line")
    if cost <= 0:
        raise ValueError("BKS cost must be positive")
    return cost


def validate(instance_dir: Path, report_path: Path) -> int:
    instances = {path.stem: path for path in instance_dir.glob("*.vrp")}
    bks_by_ext = {
        extension: {path.stem: path for path in instance_dir.glob(f"*{extension}")}
        for extension in (".bks", ".sol")
    }
    bks = bks_by_ext[".bks"] or bks_by_ext[".sol"]
    bks_extension = ".bks" if bks_by_ext[".bks"] else ".sol"

    rows = []
    errors = []
    node_counts = Counter()
    for instance_id, instance_path in sorted(instances.items()):
        bks_path = bks.get(instance_id)
        row = {
            "instancia_id": instance_id,
            "instance_file": instance_path.name,
            "bks_file": bks_path.name if bks_path else "",
            "bks_extension": bks_extension if bks_path else "",
            "customers": "",
            "capacity": "",
            "bks_cost": "",
            "status": "valid",
            "error": "",
        }
        try:
            customers, capacity = read_header(instance_path)
            row["customers"] = customers
            row["capacity"] = capacity
            node_counts[customers] += 1
            if bks_path is None:
                raise ValueError(f"missing paired {bks_extension} file")
            row["bks_cost"] = read_bks_cost(bks_path)
        except (OSError, UnicodeError, ValueError) as error:
            row["status"] = "invalid"
            row["error"] = str(error)
            errors.append(f"{instance_id}: {error}")
        rows.append(row)

    orphan_bks = sorted(set(bks) - set(instances))
    for instance_id in orphan_bks:
        errors.append(f"orphan {bks_extension}: {instance_id}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(row["status"] == "valid" for row in rows)
    print(f"instances={len(instances)}")
    print(f"bks_files={len(bks)} (extension={bks_extension if bks else 'none'})")
    print(f"valid={valid} invalid={len(rows) - valid} orphan_bks={len(orphan_bks)}")
    print(f"customer_counts={dict(sorted(node_counts.items()))}")
    print(f"report={report_path}")
    if errors:
        print("errors:")
        print("\n".join(errors[:20]))
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more")
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/input_validation.csv"))
    args = parser.parse_args()
    return validate(args.instances_dir, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
