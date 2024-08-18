# Reach cube
#python source/standalone/workflows/skrl/train.py --task Isaac-Reach-Franka-v0 --headless --enable_cameras --video --video_length 100 --video_interval 500
#python source/standalone/workflows/rsl_rl/train.py --task Isaac-Reach-Franka-v0 --headless --enable_cameras --video --video_length 100 --video_interval 500
#python source/standalone/workflows/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --headless --video --video_length 200

# Lift cube
#python source/standalone/workflows/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --enable_cameras --video --video_length 100 --video_interval 500
#python source/standalone/workflows/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-v0 --headless --video --video_length 500
#python source/standalone/workflows/skrl/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --enable_cameras --video --video_length 100 --video_interval 500 --max_iterations 1500
#python source/standalone/workflows/skrl/play.py --task Isaac-Lift-Cube-Franka-v0 --headless --video --video_length 500

# Lift cube (RGB)
python source/standalone/workflows/rsl_rl/train_rgb.py --task Isaac-Lift-Cube-Franka-v0-RGB-rsl_rl --num_envs 12 --headless --enable_cameras --video --video_length 100 --video_interval 500
#python source/standalone/workflows/rsl_rl/train_rgb.py --task Isaac-Lift-Cube-Franka-v0-RGB-rsl_rl --headless

#python source/standalone/workflows/rl_games/train.py --task Isaac-Lift-Cube-Franka-v0 --headless --enable_cameras --video --video_length 100 --video_interval 500
#python source/standalone/workflows/rl_games/play.py --task Isaac-Lift-Cube-Franka-v0 --headless --enable_cameras --video --video_length 500