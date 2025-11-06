import numpy as np
import torch
from torch import nn
from torch.nn import Module

from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from hilo.loss.focal_loss import FocalLoss


class OptimalMatchLoss(Module):
    def __init__(self, cfg):
        super(OptimalMatchLoss, self).__init__()

        self.reg_loss_cfg = cfg.get("regression_loss", None)
        self._init_regression_loss(self.reg_loss_cfg)

        self.cls_loss_cfg = cfg.get("classification_loss", None)
        self._init_classification_loss(self.cls_loss_cfg)

        self.cost_matrix_cfg = cfg.get("cost_matrix", {})

    def _init_regression_loss(self, reg_cfg):
        if reg_cfg["type"] == "l2":
            self.reg_loss = nn.MSELoss(**reg_cfg.get("kwargs", {}))
        elif reg_cfg["type"] == "l1":
            self.reg_loss = nn.L1Loss(**reg_cfg.get("kwargs", {}))
        else:
            raise ValueError(f"Unsupported regression loss type: {reg_cfg['type']}")

    def _init_classification_loss(self, cls_cfg):
        if cls_cfg["type"] == "cross_entropy":
            self.cls_loss = nn.CrossEntropyLoss(**cls_cfg.get("kwargs", {}))
        elif cls_cfg["type"] == "focal":
            self.cls_loss = FocalLoss(cls_cfg)

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
        )  # (B, K)

        assigned_batch_indices = assigned_batch_indices[
            assigned_target_mask
        ]  # filter out predictions that where assigned to padded targets
        assigned_pred_indices = assigned_pred_indices[
            assigned_target_mask
        ]  # filter out predictions that where assigned to padded targets
        assigned_target_indices = assigned_target_indices[assigned_target_mask]

        # get indices of unassigned predictions/targets
        pred_assigned_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        pred_assigned_mask = torch.index_put(
            pred_assigned_mask,
            (assigned_batch_indices, assigned_pred_indices),
            torch.ones_like(assigned_pred_indices, dtype=torch.bool),
        )

        tgt_assigned_mask = torch.zeros((B, M), dtype=torch.bool, device=device)
        tgt_assigned_mask = torch.index_put(
            tgt_assigned_mask,
            (assigned_batch_indices, assigned_target_indices),
            torch.ones_like(assigned_target_indices, dtype=torch.bool),
        )
        tgt_assigned_mask = tgt_assigned_mask

        all_pred_indices = torch.stack(
            (
                torch.arange(B, device=device).unsqueeze(-1).expand(-1, N),
                torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
            ),
            dim=-1,
        )  # (B, N, 2)
        all_target_indices = torch.stack(
            (
                torch.arange(B, device=device).unsqueeze(-1).expand(-1, M),
                torch.arange(M, device=device).unsqueeze(0).expand(B, -1),
            ),
            dim=-1,
        )  # (B, M, 2)

        unassigned_pred_mask = ~pred_assigned_mask
        unassigned_target_mask = (
            ~tgt_assigned_mask & target_mask
        )  # only consider valid targets
        unassigned_pred_indices = all_pred_indices[
            unassigned_pred_mask
        ]  # (num_unassigned_preds, 2)
        unassigned_target_indices = all_target_indices[
            unassigned_target_mask
        ]  # (num_unassigned_targets, 2)

        # assert assigned + unassigned = total
        assert (
            assigned_pred_indices.shape[0] + unassigned_pred_indices.shape[0] == B * N
        ), "total predictions do not match"
        assert (
            assigned_target_indices.shape[0] + unassigned_target_indices.shape[0]
            == torch.sum(target_mask).item()
        ), "total targets do not match"
        assert torch.all(
            pred_assigned_mask + unassigned_pred_mask == 1
        ).item(), "prediction assignment masks do not match"
        assert torch.all(
            tgt_assigned_mask + unassigned_target_mask == target_mask
        ).item(), "target assignment masks do not match"
        assert (
            target_mask[tgt_assigned_mask].all().item()
        ), "assigned targets must be valid"
        assert (
            target[unassigned_target_mask].all().item()
        ), "unassigned targets must be valid"

        asso_results = {
            True: {
                "prediction_indices": torch.stack(
                    (assigned_batch_indices, assigned_pred_indices), dim=-1
                ),
                "target_indices": torch.stack(
                    (assigned_batch_indices, assigned_target_indices), dim=-1
                ),
                "prediction": {k: v[pred_assigned_mask] for k, v in pred.items()},
                "target": target[tgt_assigned_mask],
                # "target_mask": target_mask[tgt_assigned_mask],
            },
            False: {
                "prediction_indices": unassigned_pred_indices,
                "target_indices": unassigned_target_indices,
                "prediction": {k: v[unassigned_pred_mask] for k, v in pred.items()},
                "target": target[unassigned_target_mask],
                # "target_mask": target_mask[unassigned_target_mask],
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
                center_loss + size_loss + vel_loss + heading_loss
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
            center_cost = torch.cdist(pred_centers, target_centers, p=2)  # (B, N, M)
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
            size_cost = torch.cdist(pred_sizes, target_sizes, p=2)
            weighted_size_cost = (
                self.cost_matrix_cfg["size_distance_weight"] * size_cost
            )
            cost_matrix = torch.where(
                target_valid, cost_matrix + weighted_size_cost, pad_value
            )
            costs["size_cost"] = size_cost
            costs["weighted_size_cost"] = weighted_size_cost

        if self.cost_matrix_cfg["velocity_distance_weight"] > 0.0:
            pred_velocities = pred["velocities"]  # (B, N, 2)
            target_velocities = target[..., 4:6]  # (B, M, 2)
            velocity_cost = torch.cdist(pred_velocities, target_velocities, p=2)
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
            # num_classes = pred_class_logits.size(-1)
            # tgt.clamp_(min=0, max=num_classes - 2)  # (B, M)
            # raise ValueError("There are way more classes than expected. Ther must be an mapping in the loading functionality of original HiLO.")

            # −log P(class) per (N,M) pair
            log_probs = F.log_softmax(pred_class_logits, dim=-1)  # (B, N, C)
            log_probs_exp = log_probs.unsqueeze(2).expand(-1, -1, M, -1)  # (B, N, M, C)
            cls_cost = -log_probs_exp.gather(  # (B, N, M, 1)
                dim=-1,
                index=tgt.unsqueeze(1)
                .expand(-1, pred_class_logits.size(1), -1)
                .unsqueeze(-1),
            ).squeeze(
                -1
            )  # (B, N, M)

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
