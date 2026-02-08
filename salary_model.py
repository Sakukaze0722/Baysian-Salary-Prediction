"""
Naive Bayes model and Variable Elimination for Bayesian networks.
"""

from bn_core import Variable, Factor, BN
import csv


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object.
    :return: a new Factor object resulting from normalizing factor.
    '''
    new_factor = Factor(factor.name, factor.get_scope())

    total = sum(factor.values)

    if total == 0:
        new_factor.values = list(factor.values)
        return new_factor

    new_factor.values = [v / total for v in factor.values]

    return new_factor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    '''
    old_scope = factor.get_scope()

    if variable not in old_scope:
        new_factor = Factor(factor.name, old_scope)
        new_factor.values = list(factor.values)
        return new_factor

    new_scope = [v for v in old_scope if v != variable]
    new_factor = Factor(factor.name, new_scope)

    pos_in_new = {var: i for i, var in enumerate(new_scope)}

    def recurse(idx, current_assignment):
        """Helper function, each handle method will have their own
        recurse function inside since after trying to define a global
        helper would be more difficult. Each helper has nearly same logics"""
        if idx == len(new_scope):
            full_assignment = []
            for v in old_scope:
                if v == variable:
                    full_assignment.append(value)
                else:
                    full_assignment.append(current_assignment[pos_in_new[v]])

            val = factor.get_value(list(full_assignment))

            new_factor.add_values([list(current_assignment) + [val]])
            return

        v = new_scope[idx]
        for val in v.domain():
            current_assignment.append(val)
            recurse(idx + 1, current_assignment)
            current_assignment.pop()

    recurse(0, [])

    return new_factor



def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    old_scope = factor.get_scope()

    if variable not in old_scope:
        new_factor = Factor(factor.name, old_scope)
        new_factor.values = list(factor.values)
        return new_factor

    new_scope = [v for v in old_scope if v != variable]
    new_factor = Factor(factor.name, new_scope)

    if len(new_scope) == 0:
        total = 0.0
        for val in variable.domain():
            total += factor.get_value([val])
        new_factor.add_values([[total]])
        return new_factor

    pos_in_new = {var: i for i, var in enumerate(new_scope)}

    def recurse(idx, current_assignment):
        """Helper recurse similar to restrict"""
        if idx == len(new_scope):
            total = 0.0
            for val in variable.domain():
                full_assignment = []
                for v in old_scope:
                    if v == variable:
                        full_assignment.append(val)
                    else:
                        full_assignment.append(current_assignment[pos_in_new[v]])
                total += factor.get_value(list(full_assignment))

            new_factor.add_values([list(current_assignment) + [total]])
            return

        v = new_scope[idx]
        for val in v.domain():
            current_assignment.append(val)
            recurse(idx + 1, current_assignment)
            current_assignment.pop()

    recurse(0, [])

    return new_factor


def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    if not factor_list:
        unit_factor = Factor("unit", [])
        unit_factor.values = [1.0]
        return unit_factor

    new_scope = []
    for f in factor_list:
        for v in f.get_scope():
            if v not in new_scope:
                new_scope.append(v)

    new_factor = Factor("product", new_scope)
    scope = new_factor.get_scope()

    def recurse(idx, current_assignment):
        """Helper recurse similar to recurse"""
        if idx == len(scope):
            prod = 1.0
            for f in factor_list:
                f_scope = f.get_scope()
                vals_for_f = []
                for v in f_scope:
                    j = scope.index(v)
                    vals_for_f.append(current_assignment[j])
                prod *= f.get_value(list(vals_for_f))

            new_factor.add_values([list(current_assignment) + [prod]])
            return

        var = scope[idx]
        for val in var.domain():
            current_assignment.append(val)
            recurse(idx + 1, current_assignment)
            current_assignment.pop()

    recurse(0, [])

    return new_factor



def ve(bayes_net, var_query, varlist_evidence):
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the
    evidence provided by varlist_evidence.

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param varlist_evidence: the evidence variables. Each evidence variable has
                         its evidence set to a value from its domain
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given
             the settings of the evidence variables.

    '''
    if var_query in varlist_evidence:
        val = var_query.get_evidence()
        result = Factor(f"P({var_query.name}|e)", [var_query])
        rows = []
        for v in var_query.domain():
            prob = 1.0 if v == val else 0.0
            rows.append([v, prob])
        result.add_values(rows)
        return result

    factors = bayes_net.factors()

    for ev in varlist_evidence:
        ev_val = ev.get_evidence()
        new_factors = []
        for f in factors:
            if ev in f.get_scope():
                new_factors.append(restrict(f, ev, ev_val))
            else:
                new_factors.append(f)
        factors = new_factors

    elimination_vars = [
        v for v in bayes_net.variables()
        if (v is not var_query) and (v not in varlist_evidence)
    ]

    for Z in elimination_vars:
        with_Z = []
        without_Z = []
        for f in factors:
            if Z in f.get_scope():
                with_Z.append(f)
            else:
                without_Z.append(f)

        if not with_Z:
            factors = without_Z
            continue


        combined = multiply(with_Z)
        reduced = sum_out(combined, Z)

        factors = without_Z + [reduced]

    if not factors:
        result = Factor(f"P({var_query.name}|e)", [var_query])
        rows = [[val, 1.0] for val in var_query.domain()]
        result.add_values(rows)
    elif len(factors) == 1:
        result = factors[0]
    else:
        result = multiply(factors)

    result = normalize(result)
    return result



## The order of these domains is consistent with the order of the columns in the data set.
salary_variable_domains = {
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

salary_variable=Variable("Salary", ['<50K', '>=50K'])

def naive_bayes_model(data_file, variable_domains=salary_variable_domains, class_var=salary_variable):
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

    @return a BN that is a Naive Bayes model and which represents the given data set.
    '''
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



# Re-export explore for backward compatibility; implementation lives in fairness_metrics
from fairness_metrics import explore

if __name__ == "__main__":
    # Use tiny dataset for quick runs; adult-train.csv for full training
    train_file = "adult-train_tiny.csv"

    # Build model and run fairness analysis
    bn = naive_bayes_model(train_file)

    for q in range(1, 7):
        value = explore(bn, q)
        print(f"Q{q}: {value:.2f}%")

