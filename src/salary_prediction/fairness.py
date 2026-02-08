"""
Fairness Analysis Module for Bayesian Salary Prediction.

Evaluates the Naive Bayes model across demographic parity, separation,
and sufficiency metrics on the UCI Adult dataset.
"""

import csv
import os

from .inference import ve


def _find_test_data_path(default_name="adult-test.csv"):
    """Resolve test data path, checking data/ and project root."""
    candidates = [
        os.path.join("data", default_name),
        default_name,
        os.path.join(os.path.dirname(__file__), "..", "..", "data", default_name),
        os.path.join(os.path.dirname(__file__), "..", "..", default_name),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path
    raise FileNotFoundError(
        f"Test data file '{default_name}' not found. "
        "Place it in project root or data/ directory."
    )


def load_test_data(test_data_path=None):
    """
    Load test dataset from CSV.

    :param test_data_path: Path to CSV file. If None, auto-detects.
    :return: Tuple of (rows list, headers list, column index dict).
    """
    path = test_data_path or _find_test_data_path()
    input_data = []
    with open(path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        headers = [h.strip().lstrip("\ufeff") for h in headers]
        for row in reader:
            if not row:
                continue
            input_data.append(row)
    idx = {name: i for i, name in enumerate(headers)}
    return input_data, headers, idx


def compute_p_ge50(bayes_net, row, idx, include_gender=False):
    """
    Compute P(Salary >= $50K | Evidence) for a single row.

    :param bayes_net: Trained BN (Naive Bayes model).
    :param row: Data row as list.
    :param idx: Column name -> index mapping.
    :param include_gender: If True, add Gender to evidence.
    :return: Probability in [0, 1].
    """
    evidence_names = ["Work", "Education", "Occupation", "Relationship"]
    if include_gender:
        evidence_names = evidence_names + ["Gender"]

    varlist_evidence = []
    for name in evidence_names:
        var = bayes_net.get_variable(name)
        val = row[idx[name]]
        var.set_evidence(val)
        varlist_evidence.append(var)

    salary_var = bayes_net.get_variable("Salary")
    factor = ve(bayes_net, salary_var, varlist_evidence)
    return factor.get_value([">=50K"])


def run_fairness_analysis(bayes_net, test_data_path=None):
    """
    Run full fairness analysis and return all six metrics.

    :param bayes_net: Trained Naive Bayes BN.
    :param test_data_path: Optional path to test CSV.
    :return: Dict mapping question number (1-6) to percentage.
    """
    input_data, _, idx = load_test_data(test_data_path)

    def is_female(row):
        return row[idx["Gender"]] == "Female"

    def is_male(row):
        return row[idx["Gender"]] == "Male"

    def p_ge_50(row, include_gender=False):
        return compute_p_ge50(bayes_net, row, idx, include_gender)

    results = {}

    # Q1: % women predicted >= $50K
    women_rows = [r for r in input_data if is_female(r)]
    if women_rows:
        count = sum(1 for r in women_rows if p_ge_50(r, include_gender=False) > 0.5)
        results[1] = 100.0 * count / len(women_rows)
    else:
        results[1] = 0.0

    # Q2: % men predicted >= $50K
    men_rows = [r for r in input_data if is_male(r)]
    if men_rows:
        count = sum(1 for r in men_rows if p_ge_50(r, include_gender=False) > 0.5)
        results[2] = 100.0 * count / len(men_rows)
    else:
        results[2] = 0.0

    # Q3: % women with P(S|E) > P(S|E,Gender)
    if women_rows:
        count = sum(
            1 for r in women_rows
            if p_ge_50(r, include_gender=False) > p_ge_50(r, include_gender=True)
        )
        results[3] = 100.0 * count / len(women_rows)
    else:
        results[3] = 0.0

    # Q4: % men with P(S|E) > P(S|E,Gender)
    if men_rows:
        count = sum(
            1 for r in men_rows
            if p_ge_50(r, include_gender=False) > p_ge_50(r, include_gender=True)
        )
        results[4] = 100.0 * count / len(men_rows)
    else:
        results[4] = 0.0

    # Q5: Women positive-prediction accuracy
    positives = [r for r in women_rows if p_ge_50(r, include_gender=False) > 0.5]
    if positives:
        correct = sum(1 for r in positives if r[idx["Salary"]] == ">=50K")
        results[5] = 100.0 * correct / len(positives)
    else:
        results[5] = 0.0

    # Q6: Men positive-prediction accuracy
    positives = [r for r in men_rows if p_ge_50(r, include_gender=False) > 0.5]
    if positives:
        correct = sum(1 for r in positives if r[idx["Salary"]] == ">=50K")
        results[6] = 100.0 * correct / len(positives)
    else:
        results[6] = 0.0

    return results


def explore(bayes_net, question, test_data_path=None):
    """
    Return a single fairness metric (Q1–Q6).

    The questions:
    1. % women predicted salary >= $50K (demographic parity)
    2. % men predicted salary >= $50K (demographic parity)
    3. % women with P(S|E) > P(S|E,Gender) (separation)
    4. % men with P(S|E) > P(S|E,Gender) (separation)
    5. Women positive-prediction accuracy (sufficiency)
    6. Men positive-prediction accuracy (sufficiency)

    :param bayes_net: Trained BN.
    :param question: Integer 1–6.
    :param test_data_path: Optional path to test CSV.
    :return: Percentage in [0, 100].
    """
    results = run_fairness_analysis(bayes_net, test_data_path)
    if question not in results:
        raise ValueError(f"Question must be 1–6, got {question}")
    return results[question]
