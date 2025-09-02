def warn_overhead_cost(*args, **kwargs):
    def wrapper(func):
        omni.log.warn(f"OverheadWarning: {func.__name__} is expensive and should be avoided. Instead of getting/setting sliced data, use the whole data in the target kernel.")
        return func(*args, **kwargs)
    return wrapper
