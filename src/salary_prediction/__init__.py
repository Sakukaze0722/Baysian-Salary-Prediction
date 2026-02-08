"""
Bayesian Salary Prediction & Fairness Analysis.

A Bayesian inference system for income prediction and AI fairness analysis
on the UCI Adult dataset.
"""

from .bn_core import Variable, Factor, BN
from .inference import ve, normalize, restrict, sum_out, multiply
from .model import naive_bayes_model, salary_variable_domains, salary_variable
from .fairness import explore, run_fairness_analysis, load_test_data

__all__ = [
    "Variable",
    "Factor",
    "BN",
    "ve",
    "normalize",
    "restrict",
    "sum_out",
    "multiply",
    "naive_bayes_model",
    "salary_variable_domains",
    "salary_variable",
    "explore",
    "run_fairness_analysis",
    "load_test_data",
]

__version__ = "0.1.0"
