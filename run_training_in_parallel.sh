./run_training.sh train_state >train_state.log 2>&1 &
./run_training.sh train_rgb_and_state >train_rgb_and_state.log 2>&1 &
wait