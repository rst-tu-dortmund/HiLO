import logging
import torch
import numpy as np
from shapely.geometry import Polygon

try:
    import hilo.evaluation.metrics.iou_bev_cpp as iou_bev_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ IoU extension not available, using pure Python.")


def create_polygon(bb):
    if isinstance(bb, torch.Tensor):
        bb = bb.detach().cpu().numpy()

    center_x, center_y, length, width, v_x, v_y, hdg = bb
    sin_hdg = np.sin(hdg)
    cos_hdg = np.cos(hdg)

    x1 = center_x + length / 2 * cos_hdg - width / 2 * sin_hdg
    y1 = center_y + length / 2 * sin_hdg + width / 2 * cos_hdg
    x2 = center_x - length / 2 * cos_hdg - width / 2 * sin_hdg
    y2 = center_y - length / 2 * sin_hdg + width / 2 * cos_hdg
    x3 = center_x - length / 2 * cos_hdg + width / 2 * sin_hdg
    y3 = center_y - length / 2 * sin_hdg - width / 2 * cos_hdg
    x4 = center_x + length / 2 * cos_hdg + width / 2 * sin_hdg
    y4 = center_y + length / 2 * sin_hdg - width / 2 * cos_hdg

    return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

def rotated_iou_bev(boxes1, boxes2, boxes1_mask=None, boxes2_mask=None):
    if CPP_AVAILABLE:
        B, M, _ = boxes1.shape
        _, N, _ = boxes2.shape
        
        # Prepare inputs
        b1_c = boxes1.contiguous()
        b2_c = boxes2.contiguous()
        
        if boxes1_mask is None:
            m1 = torch.ones((B, M), dtype=torch.bool, device=boxes1.device)
        else:
            m1 = boxes1_mask.contiguous()

        if boxes2_mask is None:
            m2 = torch.ones((B, N), dtype=torch.bool, device=boxes2.device)
        else:
            m2 = boxes2_mask.contiguous()
            
        iou_matrix_cpp = iou_bev_cpp.rotated_iou_bev_cpp(b1_c, b2_c, m1, m2)
        
        # Consistency check (enabled for verification)
        if False: # Set to True to verify
             iou_matrix_py = _rotated_iou_bev_python(boxes1, boxes2, boxes1_mask, boxes2_mask)
             diff = (iou_matrix_cpp - iou_matrix_py).abs().max()
             logging.info(f"Max IoU difference (CPP vs Python): {diff}")
             if diff > 1e-4:
                 logging.warning("WARNING: large difference detected!")
        
        return iou_matrix_cpp

    return _rotated_iou_bev_python(boxes1, boxes2, boxes1_mask, boxes2_mask)

def _rotated_iou_bev_python(boxes1, boxes2, boxes1_mask=None, boxes2_mask=None):
    if len(boxes1.shape) != 3 or boxes1.shape[-1] != 7:
        raise ValueError("boxes1 must have shape (B, M, 7)")
    
    if len(boxes2.shape) != 3 or boxes2.shape[-1] != 7:
        raise ValueError("boxes2 must have shape (B, N, 7)")
    
    if not boxes1.shape[0] == boxes2.shape[0]:
        raise ValueError("boxes1 and boxes2 must have the same batch size B")
    
    B, M, _ = boxes1.shape
    _, N, _ = boxes2.shape
    
    iou_matrix = torch.zeros((B, M, N), device=boxes1.device, requires_grad=False)
    
    for b in range(boxes1.shape[0]):
        boxes1_count = boxes1[b].shape[0]
        boxes2_count = boxes2[b].shape[0]

        # If there are no predictions or no targets, create an empty 2D tensor
        if boxes1_count == 0 or boxes2_count == 0:
            iou_matrix[b] = torch.zeros(
                (boxes1_count, boxes2_count),
                dtype=torch.float32,
                device=boxes1.device,
            )
            continue

        iou_matrix[b] = torch.tensor(
            [
                [
                    (
                        0.0
                        if (boxes1_mask is not None and not boxes1_mask[b, i])
                        else (
                            0.0
                            if (boxes2_mask is not None and not boxes2_mask[b, j])
                            else (
                                lambda b1, b2: (
                                    0.0
                                    if b1.union(b2).area < 1e-2
                                    else b1.intersection(b2).area
                                    / b1.union(b2).area
                                )
                            )(
                                create_polygon(box1),
                                create_polygon(box2),
                            )
                        )
                    )
                    for j, box2 in enumerate(boxes2[b])
                ]
                for i, box1 in enumerate(boxes1[b])
            ],
            device=boxes1.device,
        )
    
    return iou_matrix