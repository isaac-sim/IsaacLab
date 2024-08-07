MODE=$1
NUM_ENVS=10
VIDEO_LEN=100
VIDEO_INTERVAL=500
PATH_TO_CHECKPOINT="./logs/skrl/franka_lift/2024-08-06_22-22-33/checkpoints/best_agent.pt"

if [ $MODE = "train" ]; then
    echo "Training"
    # (1) RGB-based training
    python source/standalone/workflows/skrl/train_rgb.py --task Isaac-Lift-Cube-Franka-v0-RGB --num_envs $NUM_ENVS \
    --headless \
    --enable_cameras \
    --video --video_length $VIDEO_LEN --video_interval $VIDEO_INTERVAL #\
    #--checkpoint /PATH/TO/model.pt

    # (2) State-based training
    #python source/standalone/workflows/skrl/train.py --task Isaac-Reach-Franka-v0 --num_envs $NUM_ENVS \
    #--headless \
    #--enable_cameras 
    #--video --video_length $VIDEO_LEN --video_interval $VIDEO_INTERVAL \
    #--checkpoint /PATH/TO/model.pt
else
    echo "Playing"
    python source/standalone/workflows/skrl/play_rgb.py --task Isaac-Lift-Cube-Franka-v0-RGB --num_envs 2 \
    --headless \
    --enable_cameras
fi