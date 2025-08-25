from isaaclab.utils import configclass
from .lee_controller import BaseLeeController

@configclass
class LeeControllerCfg:

    class_type: type[BaseLeeController] = BaseLeeController
    
    gravity: list[float] = [0.0, 0.0, -9.81]

    K_angvel_max: list[float] = [0.2, 0.2, 0.2]

    K_angvel_min: list[float] = [0.1, 0.1, 0.1]

    K_pos_max: list[float] = [3.0, 3.0, 2.0]

    K_pos_min: list[float] = [2.0, 2.0, 1.0]

    K_rot_max: list[float] = [1.2, 1.2, 0.6]

    K_rot_min: list[float] = [0.8, 0.8, 0.4]

    K_vel_max: list[float] = [3.0, 3.0, 3.0]

    K_vel_min: list[float] = [2.0, 2.0, 2.0]

    max_inclination_angle_rad: float = 1.0471975511965976

    max_yaw_rate: float = 1.0471975511965976

    num_actions: int = 4

    randomize_params: bool = False