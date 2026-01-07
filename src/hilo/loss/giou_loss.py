import torch
from torchvision.ops.boxes import generalized_box_iou, box_convert


class NonRotatedBevGIoULoss(torch.nn.Module):
    def __init__(self, cfg):
        super(NonRotatedBevGIoULoss, self).__init__()
        self.cfg = cfg
        self.reduction = cfg.get("reduction", "mean")
        
    def forward(self, boxes1, boxes2):
        """
        boxes1, boxes2: (B, N, 5) in (x_center, y_center, length, width, heading) format
        returns: scalar giou loss
        """
        giou = non_rotated_bev_giou(
            boxes1, boxes2
        ).diagonal()  # (B, N) giou values
        
        giou_loss = 1.0 - giou
        
        if self.reduction == "sum":
            giou_loss = giou_loss.sum()
            
        elif self.reduction == "mean":
            giou_loss = giou_loss.mean()
        
        
        return giou_loss

def non_rotated_bev_giou(boxes1, boxes2):
    conv_boxes1 = box_convert(
        boxes1[..., :4], in_fmt="cxcywh", out_fmt="xyxy"
    )
    
    conv_boxes2 = box_convert(
        boxes2[..., :4], in_fmt="cxcywh", out_fmt="xyxy"
    )
    
    giou = compute_generalized_iou(conv_boxes1, conv_boxes2)
    
    return giou

class AxisAlignedBevGIoULoss(torch.nn.Module):
    def __init__(self, cfg):
        super(AxisAlignedBevGIoULoss, self).__init__()
        self.cfg = cfg
        self.reduction = cfg.get("reduction", "mean")
        
    def forward(self, boxes1, boxes2):
        """
        boxes1, boxes2: (B, N, 5) in (x_center, y_center, length, width, heading) format
        returns: scalar giou loss
        """
        
        giou = axis_aligned_bev_giou(
            boxes1, boxes2
        ).diagonal()  # (B, N) giou values
        
        giou_loss = 1.0 - giou
        
        if self.reduction == "sum":
            giou_loss = giou_loss.sum()
            
        elif self.reduction == "mean":
            giou_loss = giou_loss.mean()
        
        
        return giou_loss


def axis_aligned_bev_giou(boxes1, boxes2):
    """
    boxes1, boxes2: (..., 5) in (x_center, y_center, length, width, heading) format
    returns: (...,) giou values
    """
    device = boxes1.device
    corners = torch.Tensor(
        [
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
        ]
    ).to(torch.float32).to(device)  # (4, 2)
    
    boxes1_corners = compute_rotated_corners(boxes1, corners)
    aa_boxes1 = rotated_box_to_axis_aligned_bbox(boxes1_corners)
    
    boxes2_corners = compute_rotated_corners(boxes2, corners)
    aa_boxes2 = rotated_box_to_axis_aligned_bbox(boxes2_corners)
    
    giou = compute_generalized_iou(aa_boxes1, aa_boxes2)
    
    # if len(aa_boxes1.shape) == 2 and len(aa_boxes2.shape) == 2:
    #     giou_, inter_, union_, areai_ = generalized_box_iou(aa_boxes1, aa_boxes2)
    
    return giou

def compute_generalized_iou(aa_boxes1, aa_boxes2):
    areas1 = (aa_boxes1[..., 2] - aa_boxes1[..., 0]) * (aa_boxes1[..., 3] - aa_boxes1[..., 1])
    areas2 = (aa_boxes2[..., 2] - aa_boxes2[..., 0]) * (aa_boxes2[..., 3] - aa_boxes2[..., 1])
    
    max_mins = torch.max(aa_boxes1[..., None, :2], aa_boxes2[..., None, :, :2])  # [B,N,M,2]
    min_maxs = torch.min(aa_boxes1[..., None, 2:4], aa_boxes2[..., None, :, 2:4])  # [B,N,M,2]

    wh = (min_maxs - max_mins).clamp(min=0.0)  # [B,N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [B,N,M]

    union = areas1[..., None] + areas2[..., None, :] - inter # [B,N,M]
    union = union.clamp(min=1e-6)
    
    min_mins = torch.min(aa_boxes1[..., None, :2], aa_boxes2[..., None, :, :2])  # [B,N,M,2]
    max_maxs = torch.max(aa_boxes1[..., None, 2:4], aa_boxes2[..., None, :, 2:4])  # [B,N,M,2]
    
    whi = (max_maxs - min_mins).clamp(min=1e-2)  # [B,N,M,2]
    areai = whi[..., 0] * whi[..., 1]  # [B,N,M]
    
    giou = inter/union-(areai - union) / areai
    return giou
    
def rotated_box_to_axis_aligned_bbox(box_corners):
    """
    box_cornes: (..., 4, 2) in (x, y) format
    """
    x_min = torch.min(box_corners[..., 0], dim=-1).values
    y_min = torch.min(box_corners[..., 1], dim=-1).values
    x_max = torch.max(box_corners[..., 0], dim=-1).values
    y_max = torch.max(box_corners[..., 1], dim=-1).values
    
    axis_aligned_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    
    return axis_aligned_bboxes

def compute_rotated_corners(boxes1, corners):
    boxes1_xy = boxes1[..., 0:2]
    boxes1_lw = boxes1[..., 2:4]
    
    if len(boxes1_lw.shape) == 2:
        boxes1_corners_ = corners[None, ...] * boxes1_lw[:, None, :]
    elif len(boxes1_lw.shape) == 3:
        boxes1_corners_ = corners[None, None, ...] * boxes1_lw[..., None, :]
    
    boxes1_rot = torch.stack(
        [
            torch.stack([torch.cos(boxes1[..., 4]), -torch.sin(boxes1[..., 4])], dim=-1),
            torch.stack([torch.sin(boxes1[..., 4]), torch.cos(boxes1[..., 4])], dim=-1),
        ],
        dim=-2,
    )
    
    boxes1_corners = torch.matmul(boxes1_corners_, boxes1_rot) + boxes1_xy[..., None, :]
    
    return boxes1_corners