"""
Entry point for running salary prediction and fairness analysis.

Usage:
    python -m salary_prediction
    python -m salary_prediction --train data/adult-train.csv --test data/adult-test.csv
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Salary Prediction & Fairness Analysis"
    )
    parser.add_argument(
        "--train",
        default="data/adult-train_tiny.csv",
        help="Path to training CSV (default: data/adult-train_tiny.csv)",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Path to test CSV (default: auto-detect data/adult-test.csv)",
    )
    args = parser.parse_args()

    from .model import naive_bayes_model
    from .fairness import explore

    print("Training Naive Bayes model...")
    bn = naive_bayes_model(args.train)
    print(f"Model trained on {args.train}\n")
    print("Fairness metrics:")

    for q in range(1, 7):
        value = explore(bn, q, test_data_path=args.test)
        print(f"  Q{q}: {value:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
