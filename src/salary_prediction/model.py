"""
Naive Bayes model for Bayesian salary prediction.
"""

import csv

from .bn_core import Variable, Factor, BN

# The order of these domains is consistent with the order of the columns in the data set.
SALARY_VARIABLE_DOMAINS = {
    "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    "Gender": ['Male', 'Female'],
    "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    "Salary": ['<50K', '>=50K']
}

# Backward compatibility aliases
salary_variable_domains = SALARY_VARIABLE_DOMAINS
salary_variable = Variable("Salary", ['<50K', '>=50K'])


def naive_bayes_model(data_file, variable_domains=None, class_var=None):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that represents
    the joint distribution of value assignments to variables in the given dataset.

    Remember a Naive Bayes model assumes P(X1, X2,.... XN, Class) can be represented as
    P(X1|Class) * P(X2|Class) * .... * P(XN|Class) * P(Class).

    When you generated your Bayes Net, assume that the values in the SALARY column of
    the dataset are the CLASS that we want to predict.

    Factor naming conventions (for compatibility):
    - The Salary factor is named "Salary".
    - Other factors are named "VariableName,Salary" (e.g., "Education,Salary").

    :param data_file: Path to CSV training data.
    :param variable_domains: Dict mapping column names to domain values. Defaults to SALARY_VARIABLE_DOMAINS.
    :param class_var: Variable object for the class (Salary). Defaults to salary_variable.
    :return: a BN that is a Naive Bayes model and which represents the given data set.
    '''
    variable_domains = variable_domains or SALARY_VARIABLE_DOMAINS
    class_var = class_var or salary_variable

    input_data = []
    with open(data_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)      # skip header row
        # strip whitespace + possible BOM on the first header (e.g. '\ufeffWork')
        headers = [h.strip().lstrip('\ufeff') for h in headers]
        for row in reader:
            if not row:
                continue
            input_data.append(row)

    # Reuse the provided class_var object for "Salary"
    variables = {}
    for name in headers:
        if name == class_var.name:
            v = class_var
        else:
            v = Variable(name, variable_domains[name])
        variables[name] = v

    salary_name = class_var.name
    salary_domain = variable_domains[salary_name]

    # counts for P(Salary)
    salary_counts = {s: 0 for s in salary_domain}
    # counts for P(X | Salary)
    cond_counts = {}
    for name in headers:
        if name == salary_name:
            continue
        cond_counts[name] = {
            s: {val: 0 for val in variable_domains[name]} for s in salary_domain
        }

    for row in input_data:
        row_dict = {headers[i]: row[i] for i in range(len(headers))}
        s_val = row_dict[salary_name]
        salary_counts[s_val] += 1
        for name in headers:
            if name == salary_name:
                continue
            x_val = row_dict[name]
            cond_counts[name][s_val][x_val] += 1

    total_rows = sum(salary_counts.values())

    # P(Salary)
    salary_factor = Factor(salary_name, [variables[salary_name]])
    salary_rows = []
    for s in salary_domain:
        prob = salary_counts[s] / total_rows if total_rows > 0 else 0.0
        salary_rows.append([s, prob])
    salary_factor.add_values(salary_rows)

    factors = [salary_factor]

    # P(X | Salary)
    for name in headers:
        if name == salary_name:
            continue
        var_obj = variables[name]
        f_name = f"{name},Salary"
        factor = Factor(f_name, [var_obj, variables[salary_name]])
        rows = []
        for x_val in variable_domains[name]:
            for s in salary_domain:
                denom = salary_counts[s]
                prob = (
                    cond_counts[name][s][x_val] / denom
                    if denom > 0
                    else 0.0
                )
                rows.append([x_val, s, prob])
        factor.add_values(rows)
        factors.append(factor)

    return BN("Salary Naive Bayes", list(variables.values()), factors)
