import json
import pathlib
import time

import yaml


def timefunc(name=None):
    def time_wrapper(func):
        def _wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            print(f"{name}: Function '{func.__name__}' took {(time.time() - start_time)} seconds ---")
            return res

        return _wrapper
    return time_wrapper


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


def get_config(build_info: pathlib.Path | str | list | dict) -> list | dict:
    if isinstance(build_info, str) or isinstance(build_info, pathlib.Path):
        with open(build_info, 'r') as f:
            if str(build_info).endswith('.json'):
                build_info = json.load(f)
            elif str(build_info).endswith('.yaml'):
                build_info = yaml.load(f, yaml.FullLoader)
            else:
                raise TypeError(f'Not supported config file {build_info!r}')

    return build_info
