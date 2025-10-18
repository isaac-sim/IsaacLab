import os , yaml
from easydict import EasyDict

def remove_slice(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            remove_slice(value)
        else:
            if "slice" in str(value):  
              dictionary[key] = None
    return dictionary

def load_local_cfg(cfg_dir: str, load_name: str) -> dict:
    env_cfg_yaml_path = os.path.join(cfg_dir, f"{load_name}.yaml")
    # load yaml
    with open(env_cfg_yaml_path) as yaml_in:
        cfg = yaml.load(yaml_in, Loader=yaml.Loader)

    cfg = remove_slice(cfg)
    return EasyDict(cfg)
