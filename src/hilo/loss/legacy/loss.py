import torch
import torch.nn as nn
import torch.nn.functional as F

from hilo.loss.legacy.matcher import HungarianMatcher
from einops import rearrange

from hilo.loss.legacy.utils import compute_cross_entropy_class_weights, compute_giou


class LegacyMatchingLoss(nn.Module):
    """Computes the loss for fusion task.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, box and motion params)
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg["number_of_classes"]
        self.weights = cfg["loss_weights"]
        self.matcher = HungarianMatcher(cfg["match_weights"])
        self.mse = nn.MSELoss()
        self.cfg = cfg
        self._train = True

        train_class_weights = compute_cross_entropy_class_weights(
            cfg["class_weight_computation"],
            cfg["number_of_queries"],
            cfg["num_samples"]["train"],
        )

        self._train_class_weights = torch.ones(self.num_classes + 1)
        self._train_class_weights[0] = train_class_weights[0]
        self._train_class_weights[1] = train_class_weights[1]
        self._train_class_weights[2] = train_class_weights[2]
        self._train_class_weights[3] = train_class_weights[3]
        self._train_class_weights[4] = train_class_weights[4]
        self._train_class_weights[-1] = train_class_weights[-1]
        self.register_buffer("train_class_weights", self._train_class_weights)

        class_weights = cfg["class_weights"]["val"]
        self._val_class_weights = torch.ones(self.num_classes + 1)
        self._val_class_weights[0] = class_weights[0]
        self._val_class_weights[1] = class_weights[1]
        self._val_class_weights[2] = class_weights[2]
        self._val_class_weights[3] = class_weights[3]
        self._val_class_weights[4] = class_weights[4]
        self._val_class_weights[-1] = class_weights[-1]
        self.register_buffer("val_class_weights", self._val_class_weights)

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_class(self, outputs, targets, indices):
        """Classification loss (NLL)"""
        out_class = outputs["Object_class"]
        # _, _, target_class, _ = targets.split([4, 3, 1, 1], dim=-1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        ).to("cuda")
        target_classes_o = rearrange(target_classes_o, "T C -> (T C)")
        # target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_class, indices)]).squeeze(dim=1).to('cuda')
        target_classes = torch.full(
            out_class.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=out_class.device,
        )
        target_classes[idx] = target_classes_o.long()

        if self._train:
            loss_ce = F.cross_entropy(
                out_class.transpose(1, 2), target_classes, self.train_class_weights
            )
        else:
            loss_ce = F.cross_entropy(
                out_class.transpose(1, 2), target_classes, self.val_class_weights
            )

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["BBox"][idx]
        
        srcBoxes = src_boxes[:, :4]
        srcHead = src_boxes[:, -1]
        # target_bbox, _, _, _ = targets.split([4, 3, 1, 1], dim=-1)

        # target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_bbox, indices)], dim=0)
        target_boxes = torch.cat(
            [t["BBox"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_boxes = target_boxes.to("cuda").to(torch.float32)
        num_boxes = target_boxes.shape[0]

        # target_mot = torch.cat(
        #     [t["mot"][i] for t, (_, i) in zip(targets, indices)], dim=0
        # )
        # target_mot = target_mot.to("cuda")

        target_head = torch.cat(
            [t["head"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_head = target_head.to("cuda")

        target_bboxes = torch.cat((target_boxes, target_head), dim=-1)
        targetHead = target_bboxes[:, -1].to(torch.float32)

        # loss_bbox = F.smooth_l1_loss(src_boxes, target_boxes, beta=1.0) #changed to smooth l1 loss from l1 loss
        # equal normalization
        loss_bbox = F.mse_loss(
            srcBoxes, target_boxes, reduction="mean"
        )  # changed to mse loss from l1 loss

        losses = {}
        # losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses["loss_bbox"] = loss_bbox

        # src_boxesNorm = src_boxes.sigmoid()
        # target_boxesNorm = target_boxes.sigmoid()
        # loss_giou = 1 - torch.diag(compute_giou(src_boxesNorm,target_boxesNorm))
        loss_giou = 1 - torch.diag(compute_giou(srcBoxes, target_boxes))
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # loss_head =  F.mse_loss(srcHead, targetHead)
        loss_head = torch.mean(1 - torch.cos(srcHead - targetHead))
        losses["heading"] = loss_head

        return losses

    def loss_motion(self, outputs, targets, indices):
        """Compute the losses related to the motion parameters"""
        idx = self._get_src_permutation_idx(indices)
        src_motion = outputs["Motion_params"][idx]
        # _, target_motion, _, _ = targets.split([4, 3, 1, 1], dim=-1)

        # target_mot = torch.cat([t[i] for t, (_, i) in zip(target_motion, indices)], dim=0)
        target_mot = torch.cat(
            [t["mot"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_mot = target_mot.to("cuda")
        num_obj = target_mot.shape[0]

        loss_mot = self.mse(src_motion, target_mot)
        loss_mot = loss_mot.sum() / num_obj
        losses = {"loss_motion": loss_mot}

        return losses

    def forward(self, pred, target, target_mask):
        device = pred["bboxes"].device
        
        # cast to required format
        outputs = {}
        outputs["BBox"] = pred["bboxes"][..., (0, 1, 2, 3, -1)]
        outputs["Object_class"] = pred['class_logits']
        
        targets = []
        for b_target, b_target_mask in zip(target, target_mask):
            tgt = {}
            b_target_fltrd = b_target[b_target_mask]
            tgt["labels"] = b_target_fltrd[..., 7:8].to(torch.int64)
            tgt["BBox"] = b_target_fltrd[..., (0, 1, 2, 3)].to(torch.float32)
            tgt["mot"] = b_target_fltrd[..., 4:6].to(torch.float32)
            tgt["head"] = b_target_fltrd[..., 6:7].to(torch.float32)
            targets.append(tgt)
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        indices_t = torch.cat([torch.stack((torch.full_like(idcs[0], b), *idcs), dim=-1) for b, idcs in enumerate(indices)], dim=0).to(torch.long).to(device)
        
        asso_results = self.matcher.get_association_results(indices_t, pred, target, target_mask)
        # allObjInd = set(range(20))
        # unmatched_indices = []
        # for i in range(len(indices)):
        #      unmatched = torch.tensor(list(allObjInd - set(indices[i][0].tolist())))
        #      unmatched_indices.append(unmatched)

        # Compute losses
        loss_cls = self.loss_class(outputs, targets, indices)
        loss_bbox = self.loss_boxes(outputs, targets, indices)
        # loss_motion = self.loss_motion(outputs, targets, indices)

        losses = {}
        losses.update(loss_cls)
        losses.update(loss_bbox)
        # losses.update(loss_motion)

        # log metrics to wandb
        # if self.cfg["wandb"]["enable"]:
        #     wandb.log({"loss_ce": losses['loss_ce'],
        #            "loss_bbox": losses['loss_bbox'],
        #            "loss_giou": losses['loss_giou']#,
        #         #    "loss_motion": losses['loss_motion']
        #            })

        total_loss = (
            self.weights["ce"] * losses["loss_ce"]
            + self.weights["bbox"] * losses["loss_bbox"]
            + self.weights["giou"] * losses["loss_giou"]
            + self.weights["motion"] * losses["heading"]
        )
        
        losses = {
            "total_loss": total_loss,
            "classification_loss": losses["loss_ce"],
            "bbox_loss": losses["loss_bbox"],
            "giou_loss": losses["loss_giou"],
            "heading_loss": losses["heading"],
        }
        
        return losses, None, asso_results
