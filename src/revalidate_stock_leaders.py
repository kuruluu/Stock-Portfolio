from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from stock_search_engine import evaluate_stock_config
from stock_strategy_search import (
    RESULTS_ROOT,
    config_from_record,
    load_inputs,
)


LEADERS = {
    "Overall / near-peak balanced": "b282b191dbcba770",
    "Sector-neutral": "be90c1eae5cc1a9e",
    "Time-series leader": "965a4ce961fe00e3",
    "Time-series balanced": "cb33b7644c66d97b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revalidate the four recorded stock-momentum leaders."
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--block-length", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active = json.loads(
        (RESULTS_ROOT / "active_run.json").read_text(encoding="utf-8")
    )
    run_dir = Path(active["run_dir"])
    finalist_path = run_dir / "finalists" / "results.parquet"
    finalists = pd.read_parquet(finalist_path).set_index("config_id")
    missing = sorted(set(LEADERS.values()).difference(finalists.index))
    if missing:
        raise ValueError(f"Missing recorded finalist configurations: {missing}")

    prices, _, sector_by_symbol = load_inputs()
    records: list[dict[str, object]] = []
    for role, config_id in LEADERS.items():
        source = finalists.loc[config_id].to_dict()
        source["config_id"] = config_id
        config = config_from_record(source)
        print(f"Revalidating {role}: {config_id}", flush=True)
        result = evaluate_stock_config(
            full_prices=prices,
            sector_by_symbol=sector_by_symbol,
            config=config,
            number_of_simulations=args.simulations,
            random_seed=args.seed,
            average_block_length=args.block_length,
            batch_size=250,
        )
        result.update(
            {
                "role": role,
                "source_run_id": active["run_id"],
                "source_config_id": config_id,
                "validation_seed": args.seed,
                "average_block_length": args.block_length,
                "validated_at": datetime.now().astimezone().isoformat(),
            }
        )
        records.append(result)
        print(
            f"{role}: median CAGR {float(result['cagr_median']):.2%}",
            flush=True,
        )

    output_dir = run_dir / "revalidation" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(output_dir / "results.parquet", index=False)
    frame.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "simulations_per_strategy": args.simulations,
                "average_block_length": args.block_length,
                "source_run_id": active["run_id"],
                "results": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Saved validation results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
