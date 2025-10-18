import re

joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

patterns = [
    re.compile('.*L_hip_joint'),
    re.compile('.*R_hip_joint'),
    re.compile('F[L,R]_thigh_joint'),
    re.compile('R[L,R]_thigh_joint'),
    re.compile('.*_calf_joint'),
]

def main(args):
    """Play with RSL-RL agent."""
    for pattern in patterns:
        for name in joint_names:
            if pattern.match(name):
                print(pattern, name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_flat')
    parser.add_argument("--expid", type=str, default='2025-05-21_18-23-04')
    parser.add_argument("--model_name", type=str, default='model_299.pt')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=True)
    args = parser.parse_args()
    main(args)
