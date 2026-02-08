# Bayesian Salary Prediction & Fairness Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Bayesian inference** system for income prediction and **AI fairness analysis** on the UCI Adult dataset. Built with exact probabilistic inference (Variable Elimination) and evaluated across demographic parity, separation, and sufficiency metrics.

---

## Overview

This project implements a Naive Bayesian classifier for predicting whether an individual's salary exceeds $50K, using exact probabilistic inference rather than black-box methods. Beyond prediction, it systematically analyzes model fairness across gender groups—evaluating demographic parity, conditional independence, and predictive calibration.

**Key features:**
- **Exact inference** via Variable Elimination (no sampling approximations)
- **Interpretable** probabilistic predictions
- **Fairness-aware evaluation** with multiple metrics
- **Reproducible** pipeline on the standard UCI Adult dataset

---

## Project Structure

```
├── src/
│   └── salary_prediction/    # Main package
│       ├── bn_core.py        # Bayesian network core (Variable, Factor, BN)
│       ├── inference.py      # Variable Elimination (ve, normalize, restrict, etc.)
│       ├── model.py          # Naive Bayes model
│       └── fairness.py       # Fairness analysis & metrics
├── scripts/
│   └── run_evaluation.py     # Train & evaluate script
├── tests/
├── data/
│   ├── adult-train.csv
│   ├── adult-train_tiny.csv
│   └── adult-test.csv
├── docs/
│   └── fairness_analysis.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Requirements

- Python 3.8+
- No external dependencies (standard library only)

### Setup

1. Clone the repository
2. Place data files in the `data/` directory:
   - `adult-train.csv` (or `adult-train_tiny.csv` for quick testing)
   - `adult-test.csv`
3. (Optional) Install in development mode:

```bash
pip install -e .
```

### Run

```bash
# Option 1: Run as module (recommended)
python -m salary_prediction

# Option 2: Specify train/test paths
python -m salary_prediction --train data/adult-train.csv --test data/adult-test.csv

# Option 3: Use the script directly
python scripts/run_evaluation.py
python scripts/run_evaluation.py --train data/adult-train.csv --test data/adult-test.csv
```

### Output

The script trains a Naive Bayes model and reports six fairness-related metrics:

| Q | Metric | Description |
|---|--------|-------------|
| 1 | Women predicted ≥$50K | Demographic parity (female) |
| 2 | Men predicted ≥$50K | Demographic parity (male) |
| 3 | Women: P(S\|E) > P(S\|E,G) | Separation check (female) |
| 4 | Men: P(S\|E) > P(S\|E,G) | Separation check (male) |
| 5 | Women positive-prediction accuracy | Sufficiency (female) |
| 6 | Men positive-prediction accuracy | Sufficiency (male) |

---

## Usage as a Library

```python
from salary_prediction import naive_bayes_model, explore, run_fairness_analysis

# Train model
bn = naive_bayes_model("data/adult-train_tiny.csv")

# Get a single fairness metric (Q1–Q6)
q1_value = explore(bn, question=1)

# Get all fairness metrics
results = run_fairness_analysis(bn, test_data_path="data/adult-test.csv")
```

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data   │────▶│  Naive Bayes     │────▶│  Variable       │
│  (CSV)      │     │  Model (P(X|S))  │     │  Elimination    │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Fairness       │◀────│  P(Salary│E)     │◀────│  Exact Marginal │
│  Report         │     │  per instance     │     │  P(Salary│E)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

The model factorizes as:

> P(Salary, X₁, ..., Xₙ) = P(Salary) × ∏ᵢ P(Xᵢ | Salary)

Inference uses **Variable Elimination** to compute exact posterior P(Salary | Evidence) without sampling.

---

## Fairness Analysis

This project evaluates the model across three fairness notions:

1. **Demographic Parity** (Q1–Q2): Do prediction rates differ by gender?
2. **Separation** (Q3–Q4): Does adding gender as evidence change the prediction?
3. **Sufficiency** (Q5–Q6): Among positive predictions, how accurate is the model per group?

See [docs/fairness_analysis.md](docs/fairness_analysis.md) for methodology, results, and limitations.

---

## Data

The [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) contains census-style attributes (work, education, occupation, etc.) and a binary salary label (<50K / ≥50K).

**Columns:** Work, Education, MaritalStatus, Occupation, Relationship, Race, Gender, Country, Salary

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- UCI Machine Learning Repository for the Adult dataset
