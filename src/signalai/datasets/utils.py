import json

import yaml


def get_config(build_info) -> list[dict]:
    if isinstance(build_info, str):
        with open(build_info, 'r') as f:
            if build_info.endswith('.json'):
                build_info = json.load(f)
            elif build_info.endswith('.yaml'):
                build_info = yaml.load(f, yaml.FullLoader)

    return build_info
