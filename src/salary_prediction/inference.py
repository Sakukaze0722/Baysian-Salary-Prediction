"""
Variable Elimination and factor operations for Bayesian network inference.
"""

from .bn_core import Factor


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
             of var_query. The i-th number is the probability that var_query
             is equal to its i-th value given the settings of the evidence variables.
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
