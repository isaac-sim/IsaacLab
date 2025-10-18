# Isaaclab_Parkour

Isaaclab based Parkour locomotion

Base model: [Extreme-Parkour](https://extreme-parkour.github.io/)

https://github.com/user-attachments/assets/aa9f7ece-83c1-404f-be50-6ae6a3ba3530

## How to install

isaaclab2.2.0

```
cd IsaacLab ## going to IsaacLab
```

```
https://github.com/CAI23sbP/Isaaclab_Parkour.git ## cloning this repo
```

```
cd Isaaclab_Parkour && pip3 install -e .
```

```
cd parkour_tasks && pip3 install -e .
```

## How to train policies

### 1.1. Training Teacher Policy

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless
```

### 1.2. Training Student Policy

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless
```

## How to play your policy

### 2.1. Pretrained Teacher Policy

Download Teacher Policy by this [link](https://drive.google.com/file/d/1JtGzwkBixDHUWD_npz2Codc82tsaec_w/view?usp=sharing)

### 2.2. Playing Teacher Policy

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16
```

[Screencast from 2025년 08월 16일 12시 43분 38초.webm](https://github.com/user-attachments/assets/ff1f58db-2439-449c-b596-5a047c526f1f)

### 2.3. Evaluation Teacher Policy

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 
```

### 3.1 Pretrained Student Policy

Download Student Policy by this [link](https://drive.google.com/file/d/1qter_3JZgbBcpUnTmTrexKnle7sUpDVe/view?usp=sharing)

### 3.2. Playing Student Policy

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
```

https://github.com/user-attachments/assets/82a5cecb-ffbf-4a46-8504-79188a147c40

### 3.3. Evaluation Student Policy

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 
```

## How to deploy in IsaacLab

[Screencast from 2025년 08월 20일 18시 55분 01초.webm](https://github.com/user-attachments/assets/4fb1ba4b-1780-49b0-a739-bff0b95d9b66)

### 4.1. Deployment Teacher Policy

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 
```

### 4.2. Deployment Student Policy

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 
```

## Testing your modules

```
cd parkour_test/ ## You can test your modules in here
```

## Visualize Control (ParkourViewportCameraController)

```
press 1 or 2: Going to environment

press 8: camera forward  

press 4: camera leftward   

press 6: camera rightward   

press 5: camera backward

press 0: Use free camera (can use mouse)

press 1: Not use free camera (default)
```

## How to Deploy sim2sim or sim2real

it is a future work, i will open this repo as soon as possible

* [X] sim2sim: isaaclab to mujoco
* [ ] sim2real: isaaclab to real world

see this [repo](https://github.com/CAI23sbP/go2_parkour_deploy)

### TODO list

* [X] Opening code for training Teacher model
* [X] Opening code for training Distillation
* [X] Opening code for deploying policy in IsaacLab by demo: code refer [site](https://isaac-sim.github.io/IsaacLab/main/source/overview/showroom.html)
* [ ] Opening code for deploying policy by sim2sim (mujoco)
* [ ] Opening code for deploying policy in real world

## Citation

If you use this code for your research, you **must** cite the following paper:

```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

```
Copyright (c) 2025, Sangbaek Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

The use of this software in academic or scientific publications requires
explicit citation of the following repository:

https://github.com/CAI23sbP/Isaaclab_Parkour
```

## contact us

```
sbp0783@hanyang.ac.kr
```

# Experiment Parameter

experiment_name = "unitree_go2_parkour" : Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_ppo_cfg.py UnitreeGo2ParkourStudentPPORunnerCfg

# Run

clear && python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16 --checkpoint /home/wang/IsaacLab/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_student_ppo/2025-09-18_14-46-54/model_84699.pt --enable_cameras


clear && python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16 --checkpoint /home/wang/IsaacLab/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_student_ppo/teacher_policy/model_34700.pt
