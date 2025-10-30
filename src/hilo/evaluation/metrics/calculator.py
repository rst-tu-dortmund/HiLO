from hilo.evaluation.metrics.map import NuscMAP
import logging
import numpy as np
import torch

class MetricCalculator:
    def __init__(self, metrics_cfg):
        self.cfg = metrics_cfg
        self.classes = metrics_cfg["classes"]
        self.num_classes = len(self.classes)
        
        map_cfg = metrics_cfg.get("mAP", None)
        self.map = NuscMAP(metrics_cfg["classes"], metrics_cfg["used_classes"], config=map_cfg) if map_cfg is not None else None

        self.batch_idx = 0
        self.batch_metrics = {}
        
        self.logger = logging.getLogger(__name__)
        
    def _pack_data(self, prediction, batch):
        # pack data for metric calculation
        gt_classes = batch['gt_data'][..., 7].detach().cpu().numpy().astype(np.int32) # B x M x "x" y length width v_x v_y yaw class_id ...
        gt_mask = batch['gt_mask'].detach().cpu().numpy().astype(bool)  # B x M
        
        det_classes = prediction["class_ids"].detach().cpu().numpy().astype(np.int32)  # B x N
        det_confidences = torch.gather(prediction["class_probs"], -1, prediction["class_ids"].unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()  # B x N
        
        det_mask = det_classes < self.num_classes 
        
        center_distances = torch.cdist(batch['gt_data'][..., :2], prediction["centers"]).detach().cpu().numpy()  # B x M x N

        prep_batch = {
            "num_classes": self.num_classes,
            "gt_classes": [b_gt_classes[b_gt_mask] for b_gt_classes, b_gt_mask in zip(gt_classes, gt_mask)],
            "detector_classes": [b_det_classes[b_det_mask] for b_det_classes, b_det_mask in zip(det_classes, det_mask)],
            "detector_confidences": [b_det_confidences[b_det_mask] for b_det_confidences, b_det_mask in zip(det_confidences, det_mask)],
            "center_distances": [b_center_distances[b_gt_mask, :][:, b_det_mask] for b_center_distances, b_gt_mask, b_det_mask in zip(center_distances, gt_mask, det_mask)],
        }
            
        return prep_batch
    
    def batch(self, prediction, batch):
        data = self._pack_data(prediction, batch)
        batch_metrics = {}
        if self.map is not None:
            map_batch_metrics = self.map.eval_batch(data)
            batch_metrics.update(map_batch_metrics)

        self.batch_metrics[self.batch_idx] = batch_metrics
        self.batch_idx += 1

        return batch_metrics

    def epoch(self):
        # end of epoch, combine results
        self.logger.info(f"Combining metrics from {self.batch_idx} batches.")
        epoch_metrics = {}
        if self.map is not None:
            epoch_metrics.update(self.map.combine_batches(self.batch_metrics))
        self.reset()
        
        return epoch_metrics
    
    def reset(self):
        self.batch_metrics = {}
        self.batch_idx = 0