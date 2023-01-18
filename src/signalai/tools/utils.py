import time


def timefunc(func):
    def _wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"Function '{func.__name__}' took {(time.time() - start_time)} seconds ---")
        return res

    return _wrapper


def join_dicts(*args):
    if all([i == args[0] for i in args]):
        return args[0]
    else:
        new_info = {}
        for key, value in args[0].items():
            if all([key in i for i in args]):
                if all([value == i[key] for i in args]):
                    new_info[key] = value
        return new_info
