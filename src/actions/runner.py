"""CLI runner for Chronexa Action Assistant (TextOS).

Executes action parsing dynamically across all input commands,
validates each output against strict schemas, and writes predictions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import EVALS_DIR, OUTPUTS_DIR
from src.actions.parser import ActionParser


def run_actions(commands_path: Path, output_path: Path):
    with open(commands_path, "r", encoding="utf-8") as f:
        commands = [json.loads(line) for line in f if line.strip()]

    parser = ActionParser()
    print(f"ActionParser active provider: {parser.provider}")
    predictions = []

    for idx, item in enumerate(commands, start=1):
        c_id = item["id"]
        c_text = item["command"]
        as_of = item["as_of"]

        print(f"[{idx}/{len(commands)}] Parsing action: {c_id} - '{c_text}'...")
        actions = parser.parse(c_text, as_of)
        predictions.append({"id": c_id, "actions": actions})
        if parser.provider != "none":
            time.sleep(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"Action predictions written to {output_path} ({len(predictions)} items)")


def main():
    parser = argparse.ArgumentParser(description="Chronexa Action Assistant Runner")
    parser.add_argument("--commands", default=str(EVALS_DIR / "actions_train.jsonl"), help="Path to action commands")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "actions_train_predictions.jsonl"), help="Path to output")
    args = parser.parse_args()

    run_actions(Path(args.commands), Path(args.output))


if __name__ == "__main__":
    main()
