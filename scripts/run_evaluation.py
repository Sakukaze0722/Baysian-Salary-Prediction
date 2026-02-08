#!/usr/bin/env python3
"""
Script to train the model and run fairness evaluation.

Run from project root:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --train data/adult-train.csv --test data/adult-test.csv
"""

import argparse
import os
import sys

# Add project root to path when running script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add src to path for development (package not installed)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(SRC_PATH) and SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def main():
    parser = argparse.ArgumentParser(
        description="Train Bayesian Salary model and run fairness analysis"
    )
    parser.add_argument(
        "--train",
        default="data/adult-train_tiny.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Path to test CSV (auto-detect if not specified)",
    )
    args = parser.parse_args()

    # Change to project root so relative paths work
    os.chdir(PROJECT_ROOT)

    from salary_prediction import naive_bayes_model, explore

    print("Training Naive Bayes model...")
    bn = naive_bayes_model(args.train)
    print(f"Model trained on {args.train}\n")
    print("Fairness metrics (Q1–Q6):")

    for q in range(1, 7):
        value = explore(bn, q, test_data_path=args.test)
        print(f"  Q{q}: {value:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
