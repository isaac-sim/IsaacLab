import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

TRANSFORM_READY = False

def filter_boxes(boxes, scores, lbls, score_thresh=0.1):
    ordered_idx = np.argsort(scores, axis=-1).flatten()
    scores = scores[ordered_idx, ...]
    boxes = boxes[ordered_idx, ...]
    if lbls is not None:
        lbls = lbls[ordered_idx, ...]

    indices = np.where(scores > score_thresh)
    scores = scores[indices, ...]
    boxes = boxes[indices, ...]
    if lbls is not None:
        lbls = lbls[indices, ...]

    return boxes[0], scores[0], lbls[0] if lbls is not None else None

def estimate_point_depth(roi, depth):
    raise NotImplementedError

def compute_depth_range(depth_map, object_mask):
    raise NotImplementedError



class PoseApproximator:
    def __init__(self, device, segm_checkpoint):

        self.device = device

        self.od_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.od_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

        SAM_MODEL_TYPE = "vit_h" #vit_l, vit_b
        self.segm_model = SamPredictor(sam_model_registry[SAM_MODEL_TYPE](checkpoint=segm_checkpoint))

    def __call__(self, image, query_image, depth_map=None, od_score_thresh=0.8):
        od_inputs = self.od_processor(images=image, query_images=query_image, return_tensors="pt",).to(self.device)
        with torch.no_grad():
            # (1) OD
            od_outputs = self.od_model.image_guided_detection(**od_inputs)

            # target_sizes = torch.Tensor([image.size[::-1]])
            # od_preds = self.od_processor.post_process_image_guided_detection(outputs=od_outputs, target_sizes=target_sizes, threshold=0.1)[0]
            # od_scores = od_preds["scores"].cpu().detach().numpy()
            # od_labels = None
            # od_bboxes = od_preds["boxes"].cpu().detach().numpy()

            od_logits = torch.max(od_outputs["logits"][0], dim=-1)

            od_scores = torch.sigmoid(od_logits.values).cpu().detach().numpy()
            od_labels = od_logits.indices.cpu().detach().numpy()
            od_bboxes = od_outputs["target_pred_boxes"][0].cpu().detach().numpy()

            #TODO: filter based on score thresh
            # box: A length 4 array given a box prompt to the model, in XYXY format.

            od_bboxes, od_scores, od_labels = filter_boxes(od_bboxes, od_scores, od_labels, od_score_thresh)
            W, H = image.size
            od_bboxes[..., 0::2] *= W
            od_bboxes[..., 1::2] *= H

            # norm coords: (cx, cy, w, h) -> (cx - w/2, cy - h/2, cx + w/2, cy + w2) = (x1, y1, x2, y2)
            box_center_coords = np.asarray([ np.array([ box[0], box[1] ]) for box in od_bboxes ])

            print(box_center_coords) # debug

            # (2) Segmentation
            input_labels = np.ones(box_center_coords.shape[0])
            self.segm_model.set_image(np.asarray(image))

            # The output masks in CxHxW format, where C is the number of masks, and (H, W) is the original image size. 
            segm_masks, segm_scores, segm_logits = self.segm_model.predict(
                #box=od_bboxes,
                point_coords=box_center_coords,
                point_labels=input_labels,
                multimask_output=True,
            )
            max_idx = np.argmax(segm_scores)
            segm_masks = segm_masks[max_idx:max_idx + 1, ...]
            segm_scores = segm_scores[max_idx:max_idx + 1]

            # (3) Estimate 3D position (world) using the depth
            # (segment out the depth -> depth point estimate -> deproject bbox/centroid from 2D img space to 3D world Euclidean space -> (Xw, Yw, Zw))
            # NOTE: we can detect corner for the orientation estimation
            if TRANSFORM_READY:
                K, Tw, Tr = np.eye(4), np.eye(4), np.eye(4)
                for (cx, cy, w, h) in od_bboxes:
                    print(f"(cx, cy, w, h): {cx}, {cy}, {w}, {h}")
                    xl, yt, xr, yb = cx-w//2, cy-h//2, cx+w//w, cy+h//2
                    #tl_corner = np.array([xl, yt, 1])
                    #br_corner = np.array([xr, yb, 1])
                    center = np.array([cx, cy, 1])
                    center_depth = estimate_point_depth(center, depth_map)
                    center_3d = np.linalg.pinv(K) @ (center * center_depth)
                    pos = Tr @ Tw @ center_3d # in robot's coordinate frame (cam -> world -> robot)

                    # TODO: Compute orientation if possible (detect corners)
                    min_depth, max_depth = compute_depth_range(depth_map, segm_masks)
                    corners = None
                    estimated_orientation_euler = None # from the corners in 3d, estimate orientation (world/robot?)


            # (4) Tracker (track pose temporaly)


        return (od_scores, od_labels, od_bboxes), (segm_scores, segm_masks, box_center_coords, input_labels)

    def plot_boxes(self, img, scores, boxes):
        """Plot the boxes with original coordiantes (norm)"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)#, extent=(0, 1, 1, 0))
        ax.set_axis_off()
        for score, box in zip(scores, boxes):
            cx, cy, w, h = box
            ax.plot(
                [cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
                [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], 
                "r"
            )

            ax.text(
                cx - w / 2,
                cy + h / 2 + 0.015,
                f"{score:1.2f}",
                ha="left",
                va="top",
                color="red",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "red",
                    "boxstyle": "square,pad=.3"
                }
            )
        plt.show()

    def plot_masks(self, img, scores, masks, input_coords, input_lbls):
        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([255, 0, 0, 0.6])
                color[:-1] /= 255
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_points(coords, ax, marker_size=375):
            ax.scatter(coords[0], coords[1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

        plt.figure(figsize=(10,10))
        plt.imshow(img)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            show_mask(mask, plt.gca())
        for i, (coords, lbl) in enumerate(zip(input_coords, input_lbls)):
            show_points(coords, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')    
        plt.show()