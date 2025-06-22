import copy
import random


def mutate_float(x, change_min=1.1, change_max=1.5):
    perturb_amount = random.uniform(change_min, change_max)

    # mutation direction
    new_value = x / perturb_amount if random.random() < 0.5 else x * perturb_amount
    return new_value


def mutate_float_min_1(x, **kwargs):
    new_value = mutate_float(x, **kwargs)
    new_value = max(1.0, new_value)
    return new_value


def mutate_eps_clip(x, **kwargs):
    new_value = mutate_float(x, **kwargs)
    new_value = max(0.01, new_value)
    new_value = min(0.3, new_value)
    return new_value


def mutate_mini_epochs(x, **kwargs):
    change_amount = random.randint(1, 3)
    new_value = x + change_amount if random.random() < 0.5 else x - change_amount
    new_value = max(1, new_value)
    new_value = min(12, new_value)
    return new_value


def mutate_discount(x, **kwargs):
    """Special mutation func for parameters such as gamma (discount factor)."""
    inv_x = 1.0 - x
    # very conservative, large changes in gamma can lead to very different critic estimates
    new_inv_x = mutate_float(inv_x, change_min=1.1, change_max=1.2)
    new_value = 1.0 - new_inv_x
    return new_value


def get_mutation_func(mutation_func_name):
    try:
        func = eval(mutation_func_name)
    except Exception as exc:
        print(f'Exception {exc} while trying to find the mutation func {mutation_func_name}.')
        raise Exception(f'Could not find mutation func {mutation_func_name}')

    return func


def mutate(params, mutations, mutation_rate, pbt_change_min, pbt_change_max):
    mutated_params = copy.deepcopy(params)

    for param, param_value in params.items():

        print(f'Param {param} with value {param_value} about to be mutated')

        # toss a coin whether we perturb the parameter at all
        if random.random() > mutation_rate:
            continue

        value_before = mutated_params[param]
        mutation_func_name = mutations[param]
        mutation_func = get_mutation_func(mutation_func_name)

        mutated_value = mutation_func(param_value, change_min=pbt_change_min, change_max=pbt_change_max)
        mutated_params[param] = mutated_value

        print(f'Param {param} mutated from {value_before} to value {mutated_value}')

    return mutated_params
