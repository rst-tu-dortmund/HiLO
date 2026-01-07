import numpy as np
import torch
from torchvision.ops.boxes import generalized_box_iou, box_convert


def compute_cross_entropy_class_weights(cfg, max_num_output_objs, num_samples):
    class_counts = cfg["class_counts"]
    num_objs = (
        class_counts["truck"]
        + class_counts["car"]
        + class_counts["motorcycle"]
        + class_counts["bicycle"]
        + class_counts["pedestrian"]
    )
    class_counts = np.array(
        [
            class_counts["truck"],
            class_counts["car"],
            class_counts["motorcycle"],
            class_counts["bicycle"],
            class_counts["pedestrian"],
            num_samples * max_num_output_objs - num_objs,
        ]
    )

    norm = cfg.get("normalization", "none").lower()
    if norm == "none":
        return np.ones_like(class_counts)
    elif norm == "ins":  # Inverse of Number of Samples
        weights_ins = 1 / class_counts
        weights_ins = weights_ins / np.sum(weights_ins)
        return weights_ins
    elif norm == "isns":  # Inverse Square Root of Number of Samples
        weights_isns = 1 / np.sqrt(class_counts)
        weights_isns = weights_isns / np.sum(weights_isns)
        return weights_isns
    elif norm == "ens":  # Effective Number of Samples
        beta = cfg["beta"]
        weights_ens = (1 - beta) / (1 - np.power(beta, class_counts))
        weights_ens = weights_ens / np.sum(weights_ens)
        return weights_ens
    else:
        raise ValueError(
            f"Normalization method: {cfg['normalization']} not supported for class weights computation."
        )
        
def convert_to_x1y1x2y2(bbox):
    b = box_convert(bbox, "cxcywh", "xyxy")
    return b

def compute_giou(box1, box2):
    box1_ = box_convert(box1, "cxcywh", "xyxy")
    box2_ = box_convert(box2, "cxcywh", "xyxy")
    # Calculate the GIoU
    # iou = box_iou(box1, box2)
    giou = generalized_box_iou(box1_, box2_)
    
    return giou