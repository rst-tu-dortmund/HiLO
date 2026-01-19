import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging


class F1Score:
    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            "THRESHOLDS": [
                0.5,
                0.7,
                0.8,
                0.9,
            ],  # IoU threshold required for a TP match.
            "PRINT_CONFIG": True,  # Whether to print the config information on init. Default: False.
            "VISUALIZE": False,  # Whether to enable visualization of results. Default: False.
        }
        return default_config

    def __init__(self, classes, used_classes, config=None):
        cfg = self.get_default_config()
        cfg.update(config or {})
        self.config = cfg
        self.visualize = cfg.get("VISUALIZE", False)
        self.logger = logging.getLogger("NuscMAP")

        if cfg.get("PRINT_CONFIG", False):
            self.logger.info("F1-Score Config:")
            self.logger.info(json.dumps(self.config, indent=4))

        self.classes = classes
        self.used_classes = used_classes
        self._class_id_to_name = {
            class_id: class_name for class_id, class_name in enumerate(self.classes)
        }
        self._used_class_ids = [
            self.classes.index(class_name) for class_name in self.used_classes
        ]

        self.fields = []

    def eval_batch(self, data):
        """Evaluates the batch and returns the results."""
        res = {}

        for threshold in self.config["THRESHOLDS"]:
            res[threshold] = self.eval_batch_threshold(data, threshold)

        # aggregate results
        ret_res = {
            "F1_raw": res,
            "theshold_wise_F1": {
                threshold: (
                    2
                    * res[threshold]["true_positives"]
                    / (
                        2 * res[threshold]["true_positives"]
                        + res[threshold]["false_positives"]
                        + res[threshold]["false_negatives"]
                    )
                    if (
                        2 * res[threshold]["true_positives"]
                        + res[threshold]["false_positives"]
                        + res[threshold]["false_negatives"]
                    )
                    > 0
                    else 0.0
                )
                for threshold in self.config["THRESHOLDS"]
            },
            "threshold_wise_mIoU": {
                threshold: res[threshold]["mIoU"]
                for threshold in self.config["THRESHOLDS"]
            },
            "threshold_wise_classification_accuracy": {
                threshold: (
                    res[threshold]["true_classified"]
                    / (
                        res[threshold]["true_classified"]
                        + res[threshold]["false_classified"]
                    )
                    if (
                        res[threshold]["true_classified"]
                        + res[threshold]["false_classified"]
                    )
                    > 0
                    else 0.0
                )
                for threshold in self.config["THRESHOLDS"]
            },
            "mean_F1": np.mean(
                [
                    2
                    * res[threshold]["true_positives"]
                    / (
                        2 * res[threshold]["true_positives"]
                        + res[threshold]["false_positives"]
                        + res[threshold]["false_negatives"]
                    )
                    if (
                        2 * res[threshold]["true_positives"]
                        + res[threshold]["false_positives"]
                        + res[threshold]["false_negatives"]
                    )
                    > 0
                    else 0.0
                    for threshold in self.config["THRESHOLDS"]
                ]
            ),
            "mIoU": np.mean(
                [res[threshold]["mIoU"] for threshold in self.config["THRESHOLDS"]]
            ),
            "mean_classification_accuracy": np.mean(
                [
                    res[threshold]["true_classified"]
                    / (
                        res[threshold]["true_classified"]
                        + res[threshold]["false_classified"]
                    )
                    if (
                        res[threshold]["true_classified"]
                        + res[threshold]["false_classified"]
                    )
                    > 0
                    else 0.0
                    for threshold in self.config["THRESHOLDS"]
                ]
            ),
        }

        return ret_res

    def eval_batch_threshold(self, data, threshold):
        """Evaluates the batch and returns the results for a specific threshold."""

        sum_true_positives = 0
        sum_false_positives = 0
        sum_false_negatives = 0
        sum_assigned_ious = 0.0
        sum_true_classified = 0
        sum_false_classified = 0

        for iou_matrix, det_classes, gt_classes in zip(
            data["iou"],
            data["detector_classes"],
            data["gt_classes"],
        ):
            # process each batch element
            N = len(det_classes)
            M = len(gt_classes)
            threshold_matrix = np.eye(N) * threshold
            full_iou_matrix = np.concatenate(
                [iou_matrix, threshold_matrix], axis=0
            )  # shape: (M+N, N)
            assignments = np.stack(
                linear_sum_assignment(
                    full_iou_matrix, maximize=True
                ),  # returns row_ind, col_ind
                axis=1,
            )

            pred_unassigned_mask = assignments[:, 0] >= M
            assignments[pred_unassigned_mask, 0] = -1

            gt_idcs = set(range(M))
            gt_unassigned_idcs = gt_idcs - set(assignments[:, 0])

            false_negatives = len(gt_unassigned_idcs)
            false_positives = pred_unassigned_mask.sum().item()
            true_positives = M - false_negatives

            pred_assigned_mask = ~pred_unassigned_mask

            true_assignments = assignments[pred_assigned_mask]

            assigned_ious = iou_matrix[true_assignments[:, 0], true_assignments[:, 1]]

            assigned_gt_classes = gt_classes[true_assignments[:, 0]]
            assigned_det_classes = det_classes[true_assignments[:, 1]]

            true_classified = np.sum(assigned_gt_classes == assigned_det_classes).item()
            false_classified = np.sum(
                assigned_gt_classes != assigned_det_classes
            ).item()

            sum_true_positives += true_positives
            sum_false_positives += false_positives
            sum_false_negatives += false_negatives
            sum_assigned_ious += np.sum(assigned_ious).item()
            sum_true_classified += true_classified
            sum_false_classified += false_classified

        res = {
            "threshold": threshold,
            "true_positives": sum_true_positives,
            "false_positives": sum_false_positives,
            "false_negatives": sum_false_negatives,
            "sum_assigned_ious": sum_assigned_ious,
            "mIoU": sum_assigned_ious / sum_true_positives
            if sum_true_positives > 0
            else 0.0,
            "true_classified": sum_true_classified,
            "false_classified": sum_false_classified,
        }

        return res

    def combine_batches(self, all_res):
        """Combines results from multiple batches."""
        combined_stats = {
            threshold: {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "sum_assigned_ious": 0.0,
                "true_classified": 0,
                "false_classified": 0,
            }
            for threshold in self.config["THRESHOLDS"]
        }

        for res in all_res.values():
            for threshold in self.config["THRESHOLDS"]:
                stats = res["F1_raw"][threshold]
                combined_stats[threshold]["true_positives"] += stats["true_positives"]
                combined_stats[threshold]["false_positives"] += stats["false_positives"]
                combined_stats[threshold]["false_negatives"] += stats["false_negatives"]
                combined_stats[threshold]["sum_assigned_ious"] += stats[
                    "sum_assigned_ious"
                ]
                combined_stats[threshold]["true_classified"] += stats["true_classified"]
                combined_stats[threshold]["false_classified"] += stats[
                    "false_classified"
                ]

        ret_res = {
            "threshold_wise_F1": {},
            "threshold_wise_precision": {},
            "threshold_wise_recall": {},
            "threshold_wise_mIoU": {},
            "threshold_wise_classification_accuracy": {},
        }

        f1_scores = []
        precision_scores = []
        recall_scores = []
        miou_scores = []
        acc_scores = []

        for threshold in self.config["THRESHOLDS"]:
            stats = combined_stats[threshold]
            tp = stats["true_positives"]
            fp = stats["false_positives"]
            fn = stats["false_negatives"]

            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            miou = stats["sum_assigned_ious"] / tp if tp > 0 else 0.0

            classified_sum = stats["true_classified"] + stats["false_classified"]
            acc = (
                stats["true_classified"] / classified_sum if classified_sum > 0 else 0.0
            )

            ret_res["threshold_wise_F1"][threshold] = f1
            ret_res["threshold_wise_precision"][threshold] = precision
            ret_res["threshold_wise_recall"][threshold] = recall
            ret_res["threshold_wise_mIoU"][threshold] = miou
            ret_res["threshold_wise_classification_accuracy"][threshold] = acc

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            miou_scores.append(miou)
            acc_scores.append(acc)

        ret_res["mean_F1"] = np.mean(f1_scores).item()
        ret_res["mean_precision"] = np.mean(precision_scores).item()
        ret_res["mean_recall"] = np.mean(recall_scores).item()
        ret_res["mIoU"] = np.mean(miou_scores).item()
        ret_res["mean_classification_accuracy"] = np.mean(acc_scores).item()

        ret_res = self.flatten_dict(ret_res, sep="/")

        return ret_res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        raise NotImplementedError(
            "Combine classes class averaged not yet implemented for this metric."
        )

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        raise NotImplementedError(
            "Combine classes det averaged not yet implemented for this metric."
        )

    def flatten_dict(self, d, parent_key="", sep="/"):
        """Flattens a nested dict to a flat dict."""
        items = []

        for k, v in d.items():
            new_key = parent_key + sep + str(k) if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)
