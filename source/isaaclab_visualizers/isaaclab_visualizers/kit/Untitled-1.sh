./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--enable_cameras \
--num_envs 1 \
--generation_num_trials 1 \
--input_file ./datasets/annotated_dataset_visuomotor.hdf5 \
--output_file ./datasets/generated_dataset_visuomotor.hdf5
