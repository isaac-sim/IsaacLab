#!/bin/bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset_async.py \
--enable_pinocchio \
--enable_cameras \
--rendering_mode balanced \
--task Isaac-NutPour-GR1T2-Pink-IK-Abs-Mimic-v0 \
--generation_num_trials 100 \
--num_envs 30 \
--headless \
--input_file ./datasets/annotated_dataset.hdf5 \
--output_file ./datasets/async_generated_dataset_gr1_nut_pouring_new.hdf5 \
--early_cpu_offload