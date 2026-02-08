# Fairness Analysis Report

## 1. Overview

This document describes the fairness evaluation methodology and findings for the **Bayesian Salary Prediction** model trained on the UCI Adult dataset. The model predicts whether an individual's salary exceeds $50K based on census attributes (work, education, occupation, relationship, etc.).

We evaluate fairness across three dimensions:
- **Demographic Parity**
- **Separation**
- **Sufficiency**

---

## 2. Dataset

- **Source:** [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Sensitive attribute:** Gender (Male / Female)
- **Target:** Salary (binary: <50K / ≥50K)
- **Features used as evidence:** Work, Education, Occupation, Relationship  
  (Gender is intentionally excluded from base evidence to allow separation analysis.)

---

## 3. Fairness Metrics

### 3.1 Demographic Parity (Q1, Q2)

**Definition:** The proportion of positive predictions (salary ≥ $50K) should be similar across groups.

| Metric | Question | Description |
|--------|----------|-------------|
| Q1 | % women predicted ≥ $50K | Proportion of female test instances where P(Salary=≥50K \| Evidence) > 0.5 |
| Q2 | % men predicted ≥ $50K | Proportion of male test instances where P(Salary=≥50K \| Evidence) > 0.5 |

**Interpretation:** Large gaps (e.g., Q2 ≫ Q1) indicate demographic disparity: the model predicts high salary for men far more often than for women, even on similar evidence. This may reflect historical bias in the training data or structural differences in how attributes correlate with salary.

---

### 3.2 Separation (Q3, Q4)

**Definition:** Does adding the sensitive attribute (Gender) as evidence change the predicted probability?

| Metric | Question | Description |
|--------|----------|-------------|
| Q3 | % women with P(S\|E) > P(S\|E,G) | Among women: proportion where prediction *without* gender is *higher* than prediction *with* gender |
| Q4 | % men with P(S\|E) > P(S\|E,G) | Among men: same condition |

**Interpretation:** If P(Salary \| E) > P(Salary \| E, Gender) for many instances, then including gender tends to *lower* the probability. This can indicate that the model "adjusts down" predictions when it learns gender. Low Q3/Q4 values suggest that adding gender rarely lowers the prediction in a noticeable way, but this is only one direction of the check.

---

### 3.3 Sufficiency (Q5, Q6)

**Definition:** Among those who receive a positive prediction, how often is the prediction correct?

| Metric | Question | Description |
|--------|----------|-------------|
| Q5 | Women positive-prediction accuracy | Of women with P(≥50K \| E) > 0.5, what % actually have Salary ≥ 50K? |
| Q6 | Men positive-prediction accuracy | Of men with P(≥50K \| E) > 0.5, what % actually have Salary ≥ 50K? |

**Interpretation:** Sufficiency asks whether positive predictions are equally reliable across groups. If Q6 > Q5, positive predictions are more accurate for men than for women, which can indicate unequal calibration.

---

## 4. Typical Results (Illustrative)

On the Adult dataset with a Naive Bayes model trained on `adult-train_tiny.csv` (or full `adult-train.csv`), results typically show:

- **Q1 vs Q2:** Men receive far more positive predictions than women (e.g., ~0.9% vs ~0.06% on tiny data). This indicates a **demographic parity** violation.
- **Q3–Q4:** Often near zero—adding gender rarely changes the prediction in the direction we measure.
- **Q5–Q6:** Men’s positive predictions tend to be more accurate, but sample sizes for positives are small, so conclusions are statistically weak.

*Run `python salary_model.py` for exact numbers on your setup.*

---

## 5. Fairness Assessment Summary

### What the metrics tell us

The model is **not fair** by demographic parity: it predicts salary ≥ $50K for men much more often than for women. The separation metrics (Q3–Q4) provide limited evidence that adding gender typically lowers predictions. The sufficiency metrics (Q5–Q6) suggest that positive predictions may be more accurate for men, though based on very few cases.

### Would we use this model for salary decisions?

**No.** The model:
1. Almost never predicts ≥ $50K, and does so much more often for men than women.
2. Raises serious fairness and legal concerns for any HR or pay-setting use.
3. Is trained on historical data that may embed societal bias.
4. Relies on strong Naive Bayes independence assumptions that may not hold.

We would not recommend using this model to recommend starting salaries or make pay decisions.

---

## 6. Limitations

- **Sample size:** With tiny training data, many positive predictions are rare; Q5–Q6 can be unstable.
- **Single sensitive attribute:** We only analyze Gender; Race, Age, etc. would require additional metrics.
- **One-sided separation check:** Q3–Q4 only measure P(S\|E) > P(S\|E,G); the reverse direction is not reported.
- **Dataset bias:** The Adult dataset reflects historical census patterns; fairness on this data does not imply fairness in deployment.

---

## 7. References

- UCI Adult: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
- Fairness notions: Barocas, Hardt, Narayanan, *Fairness and Machine Learning* (2023).
