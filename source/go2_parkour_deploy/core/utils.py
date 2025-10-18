from typing import List 
import numpy as np
import mujoco


OBJECT_MAP = {
    "joint": mujoco._enums.mjtObj.mjOBJ_JOINT,
    "body": mujoco._enums.mjtObj.mjOBJ_BODY,
    "actuator": mujoco._enums.mjtObj.mjOBJ_ACTUATOR,
    "sensor":mujoco._enums.mjtObj.mjOBJ_SENSOR,
}
mujoco_to_isaac =[
3, 0, 9, 
6, 4, 1, 
10, 7, 5, 
2, 11, 8
]
# isaac_to_mujoco = [
# 1, 5, 9, 
# 0, 4, 8, 
# 3, 7, 11, 
# 2, 6, 10
# ]
stand_down_joint_pos = [
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
]
isaac_to_mujoco = [
1, 5, 9, 
0, 4, 8, 
3, 7, 11, 
2, 6, 10
]


ISAAC_JOINT_NAMES = np.array([
'FL_hip_joint',#0
'FR_hip_joint',#1
'RL_hip_joint',#2
'RR_hip_joint',#3

'FL_thigh_joint',#4
'FR_thigh_joint',#5
'RL_thigh_joint',#6
'RR_thigh_joint',#7

'FL_calf_joint',#8
'FR_calf_joint',#9
'RL_calf_joint',#10
'RR_calf_joint',#11
])

UNITREE_SDK_JOINT_NAMES = np.array([
"FR_hip_joint", 
"FR_thigh_joint", 
"FR_calf_joint",
"FL_hip_joint", 
"FL_thigh_joint", 
"FL_calf_joint",
"RR_hip_joint", 
"RR_thigh_joint",
"RR_calf_joint",
"RL_hip_joint", 
"RL_thigh_joint", 
"RL_calf_joint"
])

def dict_to_list(data: dict, keys: List):
    """construct a list of values from dictionary in the order specified by keys

    arguments
    dict -- dictionary to look up keys in
    keys -- list of keys to retrieve in orer

    return list of values from dict in same order as keys
    """
    return [data.get(key) for key in keys]

def set_matching(index:List ,data: dict, regex, value):
    """set values in dict with keys matching regex

    arguments
    dict -- dictionary to set keys in
    regex -- regex to select keys that will be set
    value -- value to set keys matching regex to
    """
    
    for i,key in enumerate(data):
        if regex.match(key):
            data[key] = value
            index.append(i)
            
def dict_from_lists(keys: List, values: List):
    """construct a dict from two lists where keys and associated values appear at same index

    arguments
    keys -- list of keys
    values -- list of values in same order as keys

    return dictionary mapping keys to values
    """

    return dict(zip(keys, values))


def get_entity_name(model: mujoco.MjModel, entity_type: str, entity_id: int) -> str:
    """Gets name of an entity based on ID

    Args:
        model (mj.MjModel): model
        entity_type (str): entity type
        entity_id (int): entity id

    Returns:
        str: entity name
    """

    if entity_type == "body":
        return model.body(entity_id).name
    return mujoco.mj_id2name(model, OBJECT_MAP[entity_type], entity_id)

def get_entity_id(model: mujoco.MjModel, entity_type: str, entity_name: str) -> int:
    """Gets name of an entity based on ID

    Args:
        model (mj.MjModel): model
        entity_type (str): entity type
        entity_name (str): entity name

    Returns:
        int: entity id

    Notes:
        If the entity does not exist, returns -1
    """
    return mujoco.mj_name2id(model, OBJECT_MAP[entity_type], entity_name)

