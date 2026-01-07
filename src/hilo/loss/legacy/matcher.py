import torch
from scipy.optimize import linear_sum_assignment
from torch import device, nn
from hilo.loss.legacy.utils import compute_giou, convert_to_x1y1x2y2
from einops import rearrange


class HungarianMatcher(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def remove_padded_objects(self, target_bbox, target_cls, target_mot, target_exists):

        # Convert target_exists to boolean mask
        target_mask = target_exists.squeeze(dim=-1).bool()

        # Use the boolean mask to filter out non-existent objects
        valid_target_bbox = target_bbox[target_mask]
        valid_target_cls = target_cls[target_mask]
        valid_target_mot = target_mot[target_mask]

        return valid_target_bbox, valid_target_cls, valid_target_mot

    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["BBox"].shape[:2]
        device = outputs["BBox"].device

        # We flatten to compute the cost matrices in a batch
        out_bbox = rearrange(
            outputs["BBox"], "B Q C -> (B Q) C", B=batch_size, Q=num_queries
        )
        # out_bbox_denorm = denormalizeBBox_t(out_bbox)
        outBbox = out_bbox[:, :4].to(torch.float32)
        out_class = rearrange(
            outputs["Object_class"], "B Q C -> (B Q) C", B=batch_size, Q=num_queries
        )
        # out_motion  = rearrange(outputs["Motion_params"], 'B Q C -> (B Q) C')

        # targetBBox, targetClass, targetMot = self.remove_padded_objects(target_bbox, target_class,
        #                                                             target_motion, target_exist)
        # targetClass = targetClass.int().to('cuda')
        # targetBBox = targetBBox.to('cuda')
        # targetMot = targetMot.to('cuda')

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["BBox"] for v in targets])
        # tgt_mot = torch.cat([v["mot"] for v in targets])
        # tar_bbox = torch.cat((tgt_bbox, tgt_mot[..., -1:]), dim=-1)
        # tar_bbox_denorm = denormalizeBBox_t(tar_bbox).to("cuda").to(torch.float32)

        targetClass = rearrange(tgt_ids, "T C -> (T C)").int().to("cuda")
        targetBBox = tgt_bbox.to("cuda").to(torch.float32)
        # targetMot = tgt_mot.to('cuda')

        # Compute the classification cost.
        cost_class = -out_class[:, targetClass]

        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox[:batch_size*num_targets,:], targetBBox, p=1)
        cost_bbox = torch.cdist(outBbox, targetBBox, p=1)

        # Compute the giou cost betwen boxes
        # src_boxesNorm = out_bbox.sigmoid()
        # target_boxesNorm = targetBBox.sigmoid()
        # cost_giou = -compute_giou(src_boxesNorm, target_boxesNorm)
        cost_giou = -compute_giou(outBbox, targetBBox)

        # Compute motion parameter loss
        # cost_mot = self.mse(out_motion[:batch_size*num_targets, :], targetMot)
        # cost_mot = torch.cdist(out_motion, targetMot, p=1)

        # Final cost matrix
        C = (
            self.weights["giou"] * cost_giou
            + self.weights["bbox"] * cost_bbox
            + self.weights["ce"] * cost_class
        )  # + cost_mot
        C = C.view(
            batch_size, num_queries, -1
        ).cpu()  # push to cpu for optimizing, remove grad and convert to numpy array

        # Apply the Hungarian algorithm
        # sizes =  [torch.count_nonzero(target_exist[i]) for i in range(target_exist.shape[0])]
        # sizes =  [len(v) for v in target_class]
        sizes = [len(v["BBox"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i].detach().numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    def get_association_results(self, masked_indices, pred, target, target_mask):
        B, N, _ = pred["centers"].shape
        M = target.shape[1]
        device = pred["centers"].device

        # assigned_batch_indices = (
        #     torch.arange(B, device=device).unsqueeze(-1).expand(-1, K)
        # )
        # assigned_pred_indices = indices[:, 0, :].to(device)  # (B, K)
        # assigned_target_indices = indices[:, 1, :].to(device)  # (B, K)

        # assigned_target_mask = torch.gather(
        #     target_mask, 1, assigned_target_indices
        # )  # (B, K)

        assigned_batch_indices = masked_indices[:, 0]
        
        assigned_pred_indices = masked_indices[
            :, 1
        ]  # filter out predictions that where assigned to padded targets
        assigned_target_indices = masked_indices[
            :, 2
        ]  # filter out predictions that where assigned to padded targets

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