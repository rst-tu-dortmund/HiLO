import einops
import numpy as np
import torch
from torch import nn
from torch.nn import Module

from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from hilo.loss.focal_loss import FocalLoss
from hilo.loss.giou_loss import AxisAlignedBevGIoULoss, NonRotatedBevGIoULoss, axis_aligned_bev_giou, non_rotated_bev_giou

class OptimalMatchLoss(Module):
    def __init__(self, cfg):
        super(OptimalMatchLoss, self).__init__()

        self.reg_loss_cfg = cfg.get("regression_loss", {})
        self._init_regression_loss(self.reg_loss_cfg)

        self.cls_loss_cfg = cfg.get("classification_loss", {})
        self._init_classification_loss(self.cls_loss_cfg)
        
        self.giou_loss_cfg = cfg.get("giou_loss", {})
        self._init_giou_loss(self.giou_loss_cfg)

        self.cost_matrix_cfg = cfg.get("cost_matrix", {})

    def _init_regression_loss(self, reg_cfg):
        reg_type = reg_cfg.get("type", "l2").lower()
        if reg_type == "l2":
            self.reg_loss = nn.MSELoss(**reg_cfg.get("kwargs", {}))
        elif reg_type == "l1":
            self.reg_loss = nn.L1Loss(**reg_cfg.get("kwargs", {}))
        else:
            raise ValueError(f"Unsupported regression loss type: {reg_type}")

    def _init_classification_loss(self, cls_cfg):
        cls_weights = self.get_class_weights(cls_cfg.get("weighting", {}))
        cls_type = cls_cfg.get("type", "cross_entropy").lower()
        if cls_type == "cross_entropy":
            self.cls_loss = nn.CrossEntropyLoss(**cls_cfg.get("kwargs", {}), weight=cls_weights)
        elif cls_type == "focal":
            self.cls_loss = FocalLoss(cls_cfg)
        else:
            raise ValueError(f"Unsupported classification loss type: {cls_type}")
    
    def get_class_weights(self, weighting_cfg):
        type = weighting_cfg.get("type", "none").lower()
        if type == "none":
            return None
        
        class_counts_dict = weighting_cfg["class_counts"]
        
        class_counts = [
            class_counts_dict[cls] for cls in weighting_cfg["classes"]
        ]
        
        num_objs = sum(class_counts)
        
        class_counts.append(weighting_cfg["number_of_samples"] * weighting_cfg["number_of_predicted_objects"] - num_objs)
        class_counts = torch.tensor(class_counts, dtype=torch.float32)        
        
        if type == "ins": # Inverse Number of Samples
            weights = 1.0 / (class_counts + 1)
            weights = weights / weights.sum()
            return weights

        elif type == "isns": # Inverse Square Root Number of Samples
            weights = 1.0 / torch.sqrt(class_counts + 1)
            weights = weights / weights.sum()
            return weights
        
        elif type == "ens": # Effective Number of Samples
            beta = weighting_cfg.get("beta", 0.9999)
            effective_num = 1.0 - torch.pow(beta, class_counts)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            weights = weights / weights.sum()
            return weights
        
        else:
            raise ValueError(f"Unsupported classification weighting type: {type}")
    
    def _init_giou_loss(self, giou_cfg):
        giou_type = giou_cfg.get("type", "axis_aligned_bev_giou").lower()
        if giou_type == "axis_aligned_bev_giou":
            self.giou_loss = AxisAlignedBevGIoULoss(giou_cfg)
        elif giou_type == "non_rotated_bev_giou":
            self.giou_loss = NonRotatedBevGIoULoss(giou_cfg)
        else:
            raise ValueError(f"Unsupported giou loss type: {giou_type}")

    def _get_association_results(self, indices, pred, target, target_mask):
        B, N, _ = pred["centers"].shape
        M = target.shape[1]
        K = indices.shape[2]
        device = pred["centers"].device

        assigned_batch_indices = (
            torch.arange(B, device=device).unsqueeze(-1).expand(-1, K)
        )
        assigned_pred_indices = indices[:, 0, :].to(device)  # (B, K)
        assigned_target_indices = indices[:, 1, :].to(device)  # (B, K)

        assigned_target_mask = torch.gather(
            target_mask, 1, assigned_target_indices
        )  # (B, K) - True where assignment is to a Valid Target

        # 1. Filter assignments to keep only Valid Matches
        # We perform valid/padding filtering here to get the list of TP pairs
        valid_batch_indices = assigned_batch_indices[assigned_target_mask]
        valid_pred_indices = assigned_pred_indices[assigned_target_mask]
        valid_target_indices = assigned_target_indices[assigned_target_mask]

        # 2. Extract Matched Pairs explicitly (Preserving Order)
        # Using simple integer array indexing preserves the pairwise alignment from LSA
        res_predictions = {
            k: v[valid_batch_indices, valid_pred_indices] for k, v in pred.items()
        }
        res_targets = target[valid_batch_indices, valid_target_indices]

        # 3. Identify Unassigned Predictions (Background)
        # Create a mask of ALL predictions
        pred_assigned_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        # Mark predictions that were assigned to a VALID target as "assigned"
        pred_assigned_mask = torch.index_put(
            pred_assigned_mask,
            (valid_batch_indices, valid_pred_indices),
            torch.ones_like(valid_pred_indices, dtype=torch.bool),
        )
        
        # Unassigned = Everything else (including preds assigned to padding)
        unassigned_pred_mask = ~pred_assigned_mask

        # 4. Extract Unassigned Predictions (Boolean masking is fine here as order doesn't matter for background)
        # We don't need targets for background (they are generated as 'no-object' label later)
        unassigned_predictions = {
            k: v[unassigned_pred_mask] for k, v in pred.items()
        }
        
        # Usually we don't need explicit unmatched targets because we don't supervise them directly
        # (they are just ignored/missed). But for completeness relative to legacy structure:
        tgt_assigned_mask = torch.zeros((B, M), dtype=torch.bool, device=device)
        tgt_assigned_mask = torch.index_put(
            tgt_assigned_mask,
            (valid_batch_indices, valid_target_indices),
            torch.ones_like(valid_target_indices, dtype=torch.bool),
        )
        unassigned_target_mask = ~tgt_assigned_mask & target_mask
        unassigned_targets = target[unassigned_target_mask]

        asso_results = {
            True: {
                "prediction_indices": torch.stack(
                    (valid_batch_indices, valid_pred_indices), dim=-1
                ),
                "target_indices": torch.stack(
                    (valid_batch_indices, valid_target_indices), dim=-1
                ),
                "prediction": res_predictions,
                "target": res_targets,
            },
            False: {
                "prediction_indices": torch.nonzero(unassigned_pred_mask),
                "target_indices": torch.nonzero(unassigned_target_mask),
                "prediction": unassigned_predictions, 
                "target": unassigned_targets,
            },
        }

        return asso_results

    def forward(self, pred, target, target_mask):
        """
        pred: Dict
        target: (B, M, D)
        target_mask: (B, M) - True means valid
        """
        # target: x, y, length, width, vx, vy, heading, class_label, ...
        B, N, _ = pred["centers"].shape
        M = target.shape[1]
        device = pred["centers"].device

        cost_matrix, costs = self._compute_cost_matrix(pred, target, target_mask)
        indices = (
            torch.Tensor(
                np.array(
                    [
                        linear_sum_assignment(
                            cost_matrix_.detach().cpu().numpy(), False
                        )
                        for cost_matrix_ in cost_matrix
                    ]
                )
            )
            .to(torch.long)
            .to(device)
        )  # returns min(preds, targets) indices

        asso_results = self._get_association_results(indices, pred, target, target_mask)

        losses = self.aggregate_losses(asso_results)

        return losses, costs, asso_results

    def aggregate_losses(self, asso_results):
        #         0, 1,      2,     3,  5,  5,       6,           7
        # target: x, y, length, width, vx, vy, heading, class_label, ...
        device = asso_results[True]["prediction"]["centers"].device

        losses = {"total_loss": torch.tensor(0.0, device=device)}

        # regression loss for assigned predictions/targets
        if self.reg_loss is not None:
            center_loss = self.reg_loss(
                asso_results[True]["prediction"]["centers"],
                asso_results[True]["target"][..., :2],
            )
            losses["center_loss"] = center_loss

            size_loss = self.reg_loss(
                asso_results[True]["prediction"]["extents"],
                asso_results[True]["target"][..., 2:4],
            )
            losses["size_loss"] = size_loss

            vel_loss = self.reg_loss(
                asso_results[True]["prediction"]["velocities"],
                asso_results[True]["target"][..., 4:6],
            )
            losses["velocity_loss"] = vel_loss

            heading_loss = (
                1
                - F.cosine_similarity(
                    asso_results[True]["prediction"]["heading_directions"],
                    torch.cat(
                        [
                            torch.cos(asso_results[True]["target"][..., 6:7]),
                            torch.sin(asso_results[True]["target"][..., 6:7]),
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                ).mean()
            )
            heading_norm_loss = (
                (
                    1
                    - torch.cat(
                        [
                            asso_results[True]["prediction"]["heading_directions"],
                            asso_results[False]["prediction"]["heading_directions"],
                        ]
                    )
                    .pow(2)
                    .sum(dim=-1)
                )
                .pow(2)
                .mean()
            )
            heading_loss = (
                heading_loss + heading_norm_loss * 0.1
            )  # normalize heading vector
            losses["heading_loss"] = heading_loss

            losses["regression_loss"] = (
                self.reg_loss_cfg.get("center_weight", 1.0) * center_loss
                + self.reg_loss_cfg.get("size_weight", 1.0) * size_loss
                + self.reg_loss_cfg.get("velocity_weight", 1.0) * vel_loss
                + self.reg_loss_cfg.get("heading_weight", 1.0) * heading_loss
            )
            losses["total_loss"] = (
                losses["total_loss"]
                + self.reg_loss_cfg.get("weight", 1.0) * losses["regression_loss"]
            )

        # classification loss for assigned predictions/targets
        if self.cls_loss is not None:
            pred_cls = torch.cat(
                [
                    asso_results[True]["prediction"]["class_logits"],
                    asso_results[False]["prediction"]["class_logits"],
                ],
                dim=0,
            )

            if torch.any(asso_results[True]["target"][..., 7] < 0):
                raise ValueError(
                    "Found invalid class labels in assigned targets for classification loss."
                )
            
            tgt_cls = torch.cat(
                [
                    asso_results[True]["target"][..., 7].long(),
                    torch.full(
                        asso_results[False]["prediction"]["class_logits"].shape[:-1],
                        fill_value=asso_results[True]["prediction"][
                            "class_logits"
                        ].shape[-1]
                        - 1,
                        device=device,
                    ).long(),
                ],
                dim=0,
            )

            cls_loss = self.cls_loss(
                pred_cls,
                tgt_cls,
            )
            losses["classification_loss"] = cls_loss
            losses["total_loss"] = (
                losses["total_loss"] + self.cls_loss_cfg.get("weight", 1.0) * cls_loss
            )
        
        if self.giou_loss is not None:
            bev_pred_boxes = torch.cat(
                [
                    asso_results[True]["prediction"]["centers"],
                    asso_results[True]["prediction"]["extents"],
                    asso_results[True]["prediction"]["yaws"].unsqueeze(-1),
                ],
                dim=-1,
            )  # (num_assigned, 5)
            bev_target_boxes = torch.cat(
                [
                    asso_results[True]["target"][..., :2],
                    asso_results[True]["target"][..., 2:4],
                    asso_results[True]["target"][..., 6:7],
                ],
                dim=-1,
            )  # (num_assigned, 5)

            bev_giou_loss = self.giou_loss(bev_pred_boxes, bev_target_boxes)
            
            losses["bev_giou_loss"] = bev_giou_loss
            losses["total_loss"] = (
                losses["total_loss"]
                + self.giou_loss_cfg.get("weight", 1.0) * bev_giou_loss
            )

        return losses

    def _compute_cost_matrix(self, pred, target, target_mask):
        B, N, C = pred["class_logits"].shape
        M = target.shape[1]
        device = pred["class_logits"].device

        cost_matrix = torch.zeros((B, N, M), device=device)
        target_valid = target_mask.unsqueeze(1).expand(-1, N, -1)  # (B, N, M)
        pad_value = self.cost_matrix_cfg.get("invalid_target_cost", 1e6)
        cost_matrix = torch.where(target_valid, cost_matrix, pad_value)

        costs = {}

        if self.cost_matrix_cfg["center_distance_weight"] > 0.0:
            pred_centers = pred["centers"]  # (B, N, 2)
            target_centers = target[..., :2]  # (B, M, 2)
            center_cost = torch.cdist(pred_centers, target_centers, p=1)  # (B, N, M)
            weighted_center_cost = (
                self.cost_matrix_cfg["center_distance_weight"] * center_cost
            )
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_center_cost, pad_value
            )
            costs["center_cost"] = center_cost
            costs["weighted_center_cost"] = weighted_center_cost

        if self.cost_matrix_cfg["size_distance_weight"] > 0.0:
            pred_sizes = pred["extents"]  # (B, N, 2)
            target_sizes = target[..., 2:4]  # (B, M, 2)
            size_cost = torch.cdist(pred_sizes, target_sizes, p=1)
            weighted_size_cost = (
                self.cost_matrix_cfg["size_distance_weight"] * size_cost
            )
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_size_cost, pad_value
            )
            costs["size_cost"] = size_cost
            costs["weighted_size_cost"] = weighted_size_cost
            
        if self.cost_matrix_cfg.get("giou_weight", 0.0) > 0.0:
            pred_boxes = torch.cat(
                [
                    pred["centers"],
                    pred["extents"],
                    pred["yaws"].unsqueeze(-1),
                ],
                dim=-1,
            )  # (B, N, 5)
            target_boxes = torch.cat(
                [
                    target[..., :2],
                    target[..., 2:4],
                    target[..., 6:7],
                ],
                dim=-1,
            )  # (B, M, 5)

            giou_type = self.giou_loss_cfg.get("type", "axis_aligned_bev_giou").lower()
            if giou_type == "axis_aligned_bev_giou":
                bev_giou_cost = -axis_aligned_bev_giou(
                    pred_boxes, target_boxes
                )  # (B, N, M)
            elif giou_type == "non_rotated_bev_giou":
                bev_giou_cost = -non_rotated_bev_giou(
                    pred_boxes, target_boxes
                )  # (B, N, M)
            else:
                raise ValueError(f"Unsupported giou loss type: {giou_type}")

            weighted_bev_giou_cost = (
                -self.cost_matrix_cfg["giou_weight"] * bev_giou_cost
            )
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_bev_giou_cost, pad_value
            )
            costs["bev_giou_cost"] = bev_giou_cost
            costs["weighted_bev_giou_cost"] = weighted_bev_giou_cost

        if self.cost_matrix_cfg["velocity_distance_weight"] > 0.0:
            pred_velocities = pred["velocities"]  # (B, N, 2)
            target_velocities = target[..., 4:6]  # (B, M, 2)
            velocity_cost = torch.cdist(pred_velocities, target_velocities, p=1)
            weighted_velocity_cost = (
                self.cost_matrix_cfg["velocity_distance_weight"] * velocity_cost
            )
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_velocity_cost, pad_value
            )
            costs["velocity_cost"] = velocity_cost
            costs["weighted_velocity_cost"] = weighted_velocity_cost

        if self.cost_matrix_cfg["classification_weight"] > 0.0:
            pred_class_logits = pred["class_logits"]  # (B, N, num_classes)
            target_class_labels = target[..., 7].long()  # (B, M)

            # Make targets safe for gather (set invalid to 0; masked out later)
            tgt = target_class_labels.clone()
            tgt[~target_mask] = 0

            # −log P(class) per (N,M) pair
            log_probs = F.log_softmax(pred_class_logits, dim=-1)  # (B, N, C)
            cls_cost = torch.gather(-log_probs, dim=2, index=tgt.unsqueeze(1).expand(-1, N, -1))

            weighted_cls_cost = self.cost_matrix_cfg["classification_weight"] * cls_cost
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_cls_cost, pad_value
            )
            costs["classification_cost"] = cls_cost
            costs["weighted_classification_cost"] = weighted_cls_cost

        if self.cost_matrix_cfg["orientation_distance_weight"] > 0.0:
            # use offsetted cosine similarity of heading vectors as cost
            pred_headings = pred["heading_directions"]  # (B, N, 2)
            tgt_headings = torch.cat(
                [
                    torch.cos(target[..., 6:7]),  # x y length width vx vy heading ...
                    torch.sin(target[..., 6:7]),
                ],
                dim=-1,
            )  # (B, M, 2)
            heading_cost = 1.0 - nn.functional.cosine_similarity(
                pred_headings[..., None, :],  # (B, N, 1, 2)
                tgt_headings[:, None, ...],  # (B, 1, M, 2)
                dim=-1,
            )  # (B, N, M)

            weighted_heading_cost = (
                self.cost_matrix_cfg["orientation_distance_weight"] * heading_cost
            )
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_heading_cost, pad_value
            )
            costs["heading_cost"] = heading_cost
            costs["weighted_heading_cost"] = weighted_heading_cost

        return cost_matrix, costs
