# Disjoint Navigation

This folder contains code for running the disjoint navigation data generation script.  This assumes that you have collected a static manipulation dataset.


## Usage

To run the disjoint navigation replay script execute the following command.


```bash
./isaaclab.sh -p \
    scripts/imitation_learning/disjoint_navigation/replay.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --dataset="datasets/dataset_generated_g1_locomotion_teacher.hdf5" \
    --num_runs=1 \
    --lift_step=50 \
    --navigate_step=100 \
    --output_dir=datasets \
    --output_file_name=dataset_generated_disjoint_nav.hdf5
```


Please check ``replay.py`` for details on the arguments.