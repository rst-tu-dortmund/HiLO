import einops
import torch
from torch.nn import Module


class BaseModel(Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.inference_filtering_method = cfg.get(
            "inference_filtering_method", "nms_free"
        ).lower()

    def prepare_output(self, box_pred, class_pred):
        # compute yaw from sin and cos components
        yaw = torch.atan2(
            box_pred[..., -1], box_pred[..., -2]
        )  # assuming last two channels are cos(yaw) and sin(yaw)
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
        if self.inference_filtering_method == "nms_free":
            return self.nms_free_filter(output)
        elif self.inference_filtering_method == "hard":
            return self.hard_filter(output)
        else:
            raise ValueError(
                f"Unknown inference filtering method: {self.inference_filtering_method}"
            )

    def hard_filter(self, output):
        B, N, C = output["class_probs"].shape
        device = output["class_probs"].device

        cls_ids = output["class_ids"]
        is_no_object = cls_ids >= output["class_probs"].shape[-1] - 1

        scores = torch.gather(
            output["class_probs"], -1, cls_ids.to(torch.long).unsqueeze(-1)
        ).squeeze(-1)

        score_mask = scores >= self.cfg.get("inference_score_threshold", 0.1)
        keep_mask = ~is_no_object & score_mask
        output["mask"] = keep_mask

        return output

    def nms_free_filter(self, output):
        B, N, C = output["class_probs"].shape
        device = output["class_probs"].device
        ##### applying the NMS free filtering approach from "End-to-End Multi-Object Detection with Transformers"
        # getting the class probabilities without the "no object" class and individually (sigmoid)
        indv_cls_probs = torch.sigmoid(output["class_logits"][..., :-1])
        indv_cls_ids = torch.argmax(indv_cls_probs, dim=-1)  # shape: (B, N)

        indv_cls_probs_flat = einops.rearrange(
            indv_cls_probs, "b n c -> b (n c)"
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
        output_filtered["class_probs"] = indv_cls_probs[
            torch.arange(B, device=device).unsqueeze(-1), topk_obj_idx
        ]
        output_filtered["class_ids"] = indv_cls_ids[
            torch.arange(B, device=device).unsqueeze(-1), topk_obj_idx
        ]

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
