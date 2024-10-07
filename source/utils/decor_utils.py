import torch
from functools import wraps
import time

'''
Automatically convert a torch tensor to numpy arry
'''


def auto_numpy(method):
    def wrapper(*args, **kwargs):
        new_args, new_kwargs = [], {}
        for a in list(args):
            if torch.is_tensor(a):
                new_args.append(a.detach().cpu().float().numpy())
            elif isinstance(a, list):
                new_a = [x.detach().cpu().float().numpy() if torch.is_tensor(x) else x for x in a]
                new_args.append(new_a)
            else:
                new_args.append(a)
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                new_kwargs[key] = value.detach().cpu().numpy()
            else:
                new_kwargs[key] = value
        return method(*new_args, **new_kwargs)

    return wrapper


'''
Record the execution time of a function
'''


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper
