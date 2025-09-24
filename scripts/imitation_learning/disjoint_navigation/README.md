# Disjoint Navigation

This folder contains code for running the disjoint navigation data generation script.  This assumes that you have collected a static manipulation dataset.


## Usage

To run the disjoint navigation replay script execute the following command.


```bash
./isaaclab.sh -p \
    scripts/imitation_learning/disjoint_navigation/generate_navigation.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-Disjoint-Navigation" \
    --dataset="datasets/dataset_generated_g1_locomanipulation_teacher_release.hdf5" \
    --num_runs=1 \
    --lift_step=70 \
    --navigate_step=120 \
    --enable_pinocchio \
    --output_file=datasets/dataset_generated_disjoint_nav.hdf5 \
    --enable_cameras
```


Please check ``replay.py`` for details on the arguments.

To view the generated trajectories


```bash
./isaaclab.sh -p \
    scripts/imitation_learning/disjoint_navigation/display_dataset.py \
    datasets/dataset_generated_disjoint_nav.hdf5 \
    datasets/
```
