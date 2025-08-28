import h5py
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("output_dir", type=str)

args = parser.parse_args()

dataset = h5py.File(args.dataset, 'r')

demos = dataset['data'].keys()

plt.figure(figsize=(20, 20))

for i, demo in enumerate(demos):
    replay_data = dataset['data'][demo]['replay_state']
    path = replay_data['base_path']
    base_pose = replay_data['base_pose']
    object_pose = replay_data['object_pose']
    start_pose = replay_data['start_fixture_pose']
    end_pose = replay_data['end_fixture_pose']


    plt.figure(figsize=(20, 20))
    plt.plot(path[0, :, 0], path[0, :, 1], 'r-', label="Target Path")
    plt.plot(base_pose[:, 0], base_pose[:, 1], 'g--', label="Base Pose")
    plt.plot(object_pose[:, 0], object_pose[:, 1], 'b--', label="Object Pose")
    plt.legend(loc='upper right', ncol=1)
    plt.axis('equal')
    plt.savefig(f"datasets/{demo}.png")