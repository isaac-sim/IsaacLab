# Locomanipulation SDG

This folder contains code for running the locomanipulation SDG data generation script.  
This assumes that you have collected a static manipulation dataset.


## Usage

To run the locomanipulation SDG replay script execute the following command.


```bash
./isaaclab.sh -p \
    scripts/imitation_learning/locomanipulation_sdg/generate_navigation.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-Locomanipulation-SDG" \
    --dataset="datasets/dataset_generated_g1_locomanipulation_teacher_release.hdf5" \
    --num_runs=1 \
    --lift_step=70 \
    --navigate_step=120 \
    --enable_pinocchio \
    --output_file=datasets/dataset_generated_locomanipulation_sdg.hdf5 \
    --enable_cameras
```


Please check ``replay.py`` for details on the arguments.

To view the generated trajectories
