import imageio as io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import tqdm
import torch

from PIL import Image
from pose_approximator import PoseApproximator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Used device: {DEVICE}")

def run():
    root = "../data/"

    # Images inputs:
    img_path = "../data/480x640/rgb.jpg"
    img_pil = Image.open(img_path)
    img = np.asarray(img_pil)
    
    depth_path = "../data/480x640/depth.jpg"
    depth_pil = Image.open(depth_path)
    depth = np.asarray(depth_pil)

    # np.array inputs:
    RGB_INPUT_PATH = "../data/480x640/rgb.npy"
    DEPTH_INPUT_PATH = "../data/480x640/depth.npy"
    rgb_np = np.load(RGB_INPUT_PATH)[0]
    depth_np = np.load(DEPTH_INPUT_PATH)[0]
    print(f"RGB (np.ndarray): {rgb_np.mean()}+-{rgb_np.std()}, [{rgb_np.min()}, {rgb_np.max()}]")
    print(f"Depth (np.ndarray): {depth_np.mean()}+-{depth_np.std()}, [{depth_np.min()}, {depth_np.max()}]")

    query_img_paths = [
        os.path.join(root, "query_img1.jpg"),
        os.path.join(root, "query_img2.jpg"),
        os.path.join(root, "query_img3.jpg"),
        os.path.join(root, "480x640/query_img.jpg"),
    ]

    print(query_img_paths)

    query_imgs_pil = [ Image.open(img_path) for img_path in query_img_paths ]
    query_imgs = [ np.asarray(img) for img in query_imgs_pil ]

    K = torch.tensor([
        [293.1997,   0.0000, 128.0000],
        [  0.0000, 293.1997, 128.0000],
        [  0.0000,   0.0000,   1.0000]
    ], dtype=torch.float32)

    QUERY_IMG = query_imgs
    fig, axes = plt.subplots(1, len(QUERY_IMG), figsize=(10, 10))
    for i in range(len(QUERY_IMG)):
        axes[i].imshow(QUERY_IMG[i])

    SAM_CHECKPOINT = "../segmentation/sam_checkpoint/sam_vit_h_4b8939.pth"

    pose_approx = PoseApproximator(device=DEVICE, segm_checkpoint=SAM_CHECKPOINT)
    od_tuple, segm_tuple = pose_approx(image=Image.fromarray(rgb_np), query_image=query_imgs, intrinsics=K, od_score_thresh=0.2)

    if od_tuple[0] is None:
        raise("No detections")

    od_scores = od_tuple[0]
    od_lbls = od_tuple[1]
    od_boxes = od_tuple[2]

    print("Detected boxes: ", od_boxes)

    segm_scores = segm_tuple[0] 
    segm_masks = segm_tuple[1]

    print('scores: ', segm_scores.shape)
    print('masks: ', segm_masks.shape)

    binary_mask = np.zeros_like(depth_np)
    print(segm_masks.shape)
    print(od_boxes.shape)
    binary_mask[segm_masks[0]] = 1

    # extract points from within roi in terms of image coordinates
    cx, cy, w, h = od_boxes[0]
    x1, y1, x2, y2 = int(cx-w//2), int(cy-h//2), int(cx+w//2), int(cy+h//2)
    print(f"{x1},{y1},{x2},{y2}")
    crop = binary_mask[y1:y2, x1:x2]
    indices = np.where(crop == 1)
    x_indices, y_indices, _ = indices
    x_coords = x_indices + x1
    y_coords = y_indices + y1
    pnts2d = np.column_stack((x_coords, y_coords))
    pnts2d = np.concatenate([pnts2d, np.ones((pnts2d.shape[0], 1))], axis=1)
    print(pnts2d.shape)

    # deproject to 3d world coordinates: [N, 3] @ [3, 3]
    Kinv = np.linalg.pinv(K.cpu().numpy())
    print(Kinv.shape)
    pnts3d = pnts2d @ Kinv.T
    print(pnts3d.shape)

    x = pnts3d[:, 0]
    y = pnts3d[:, 1]
    z = pnts3d[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=1)
    plt.colorbar(sc)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Point Cloud')

if __name__ == "__main__":
    run()
    plt.show()