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
├── bn_core.py           # Bayesian network core (Variable, Factor, BN)
├── salary_model.py      # Naive Bayes model + Variable Elimination
├── fairness_metrics.py  # Fairness analysis & metrics
├── data/                # Data directory
│   ├── adult-train.csv
│   ├── adult-train_tiny.csv
│   └── adult-test.csv
├── docs/
│   └── fairness_analysis.md   # Fairness report & methodology
└── README.md
```

---

## Quick Start

### Requirements

- Python 3.8+
- No external dependencies (standard library only)

### Setup

1. Clone the repository
2. Place data files in the `data/` directory or project root:
   - `adult-train.csv` (or `adult-train_tiny.csv` for quick testing)
   - `adult-test.csv`
3. Run the main script:

```bash
# Train on tiny dataset (fast) and run fairness analysis
python salary_model.py
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
