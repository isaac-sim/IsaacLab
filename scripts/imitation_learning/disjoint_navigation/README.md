# Disjoint Navigation

## Usage

```bash
./isaaclab.sh -p \
    scripts/imitation_learning/disjoint_navigation/replay.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --dataset="datasets/dataset_generated_g1_locomotion_teacher.hdf5"
```