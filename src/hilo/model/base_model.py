import einops
import torch
from torch.nn import Module


class BaseModel(Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def prepare_output(self, box_pred, class_pred):
        # compute yaw from sin and cos components
        yaw = torch.atan2(
            box_pred[..., -2], box_pred[..., -1]
        )  # assuming last two channels are sin(yaw) and cos(yaw)
        # box_pred = torch.cat([box_pred[..., :-2], yaw.unsqueeze(-1)], dim=-1)
        output = {
            "centers": box_pred[..., :2],
            "extents": box_pred[..., 2:4],
            "yaws": yaw,
            "heading_directions": box_pred[..., -2:],
            "velocities": box_pred[..., 4:6],
            "bboxes": torch.cat([box_pred[..., :6], yaw.unsqueeze(-1)], dim=-1),
            "class_logits": class_pred,
            "class_probs": torch.softmax(class_pred, dim=-1),
            "class_ids": torch.argmax(class_pred, dim=-1).to(torch.int64),
        }

        if not self.training:
            output = self._filter_inference_output(output)

        return output

    def _filter_inference_output(self, output):
        B, N, C = output["class_probs"].shape
        device = output["class_probs"].device
        ##### applying the NMS free filtering approach from "End-to-End Multi-Object Detection with Transformers"
        # getting the class probabilities without the "no object" class and individually (sigmoid)
        indv_cls_probs = torch.sigmoid(output["class_probs"][..., :-1])
        indv_cls_ids = (
            torch.arange(indv_cls_probs.shape[-1], device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(B, N, 1)
        )

        indv_cls_probs_flat = einops.rearrange(
            indv_cls_probs, "b n c -> b (n c)"
        )  # shape: (B, N*(num_classes-1))

        indv_cls_ids_flat = einops.rearrange(
            indv_cls_ids, "b n c -> b (n c)"
        )  # shape: (B, N*(num_classes-1))

        topk_prob, topk_prob_idx = torch.topk(
            indv_cls_probs_flat, k=self.number_of_queries, dim=-1
        )
        topk_obj_idx = topk_prob_idx // (C - 1)

        output_filtered = {
            k: v[torch.arange(B, device=device).unsqueeze(-1), topk_obj_idx]
            for k, v in output.items()
        }

        # apply thresholding
        keep_mask = topk_prob >= self.cfg.get("inference_score_threshold", 0.1)
        output_filtered["mask"] = keep_mask

        return output_filtered

    def get_parameter_groups(self, weight_decay=0.01, bias_norm_decay=True):
        # returns parameter groups for optimizer with optional weight decay settings
        no_decay = []
        decay = []

        if bias_norm_decay:
            return [
                {
                    "params": self.parameters(),
                    "weight_decay": weight_decay,
                }
            ]
        else:
            for name, param in self.named_parameters():
                if "bias" in name or "norm" in name or "bn" in name or "Norm" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        return [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ]