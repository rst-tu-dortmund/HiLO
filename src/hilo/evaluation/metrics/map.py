import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging


class NuscMAP:
    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            "THRESHOLDS": [
                0.5,
                1,
                2,
                4,
            ],  # Center distance threshold required for a TP match.
            "PRECISION_THRESHOLD": 0.1,  # Precision threshold for AP calculation. Only above will be integrated
            "RECALL_THRESHOLD": 0.1,  # Recall threshold for AP calculation. Only below will be integrated
            "CLIPPING_MODE": "nusc",  # Clipping mode for AP calculation. Options: "nusc", "standard"
            "MATCH_STRATEGY": "optimal",  # Matching strategy. Options: "optimal", "greedy" (greedy would be nuScenes style)
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
            self.logger.info("NuscMAP Metric Config:")
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
        self._match_strategy = self.config["MATCH_STRATEGY"]
        self._clipping_mode = self.config["CLIPPING_MODE"]

    def eval_batch(self, data):
        """Evaluates the batch and returns the results."""
        res = {}

        for threshold in self.config["THRESHOLDS"]:
            res[threshold] = self.eval_batch_threshold(data, threshold)

        ret_res = {
            "classwise_map": {
                cls_id: np.mean(
                    [
                        res[threshold][cls_id]["class_ap"]
                        for threshold in self.config["THRESHOLDS"]
                    ]
                ).item()
                for cls_id in range(data["num_classes"])
                if cls_id in self._used_class_ids
            },
            "thresholdwise_map": {
                threshold: np.mean(
                    [
                        res[threshold][cls_id]["class_ap"]
                        for cls_id in range(data["num_classes"])
                        if cls_id in self._used_class_ids
                    ]
                ).item()
                for threshold in self.config["THRESHOLDS"]
            },
            "map": np.mean(
                [
                    np.mean(
                        [
                            res[threshold][cls_id]["class_ap"]
                            for threshold in self.config["THRESHOLDS"]
                        ]
                    )
                    for cls_id in range(data["num_classes"])
                    if cls_id in self._used_class_ids
                ]
            ).item(),
            "map_raw": res,
        }

        return ret_res

    def eval_batch_threshold(self, data, threshold):
        """Evaluates the batch and returns the results for a specific threshold."""

        num_classes = data["num_classes"]
        min_recall = self.config["RECALL_THRESHOLD"]

        num_res_el = int(np.round(min_recall * 100) + 1)
        res = {
            cls_id: {
                "class_id": cls_id,
                "class_ap": 0.0,
                "class_precision": np.zeros(101 - num_res_el),
                "class_recall": np.zeros(101 - num_res_el),
                "class_confidence": np.zeros(101 - num_res_el),
                "class_tp": 0,
                "class_tps": np.zeros(0),
                "class_fp": 0,
                "class_fps": np.zeros(0),
                "class_confidences": np.zeros(0),
                "class_num_gt": (np.concatenate(data["gt_classes"], axis=0) == cls_id)
                .sum()
                .item(),
            }
            for cls_id in range(num_classes)
        }

        for det_classes, det_conf, gt_classes, center_dists in zip(
            data["detector_classes"],
            data["detector_confidences"],
            data["gt_classes"],
            data["center_distances"],
        ):
            Nt = det_classes.shape[0]
            Ng = gt_classes.shape[0]

            if Nt == 0:
                # no dets in this frame
                # all GT are false negatives
                # this is handled by the class_num_gt already
                continue

            # split det conf and center_dists based on classes
            class_sorted_idcs = np.argsort(det_classes)
            class_ids, class_counts = np.unique(det_classes, return_counts=True)

            # first sort by class
            det_classes_sorted = det_classes[class_sorted_idcs]
            det_conf_sorted = det_conf[class_sorted_idcs]
            center_dists_sorted = center_dists[:, class_sorted_idcs]

            # split by class counts
            split_idcs = np.cumsum(class_counts[:-1])
            splitted_det_conf = np.split(det_conf_sorted, split_idcs)
            splitted_center_dists = np.split(center_dists_sorted, split_idcs, axis=1)

            if Ng == 0:
                # no gt in this frame
                for class_id, class_count, det_conf in zip(
                    class_ids, class_counts, splitted_det_conf
                ):
                    res[class_id]["class_fp"] += class_count
                    res[class_id]["class_fps"] = np.concatenate(
                        (res[class_id]["class_fps"], np.ones(class_count))
                    )
                    res[class_id]["class_tps"] = np.concatenate(
                        (res[class_id]["class_tps"], np.zeros(class_count))
                    )
                    res[class_id]["class_confidences"] = np.concatenate(
                        (res[class_id]["class_confidences"], det_conf)
                    )

                continue

            for class_id, class_det_conf, class_center_dists in zip(
                class_ids, splitted_det_conf, splitted_center_dists
            ):
                if self._match_strategy == "optimal":
                    class_Nt = class_det_conf.shape[0]

                    # filter GT by class
                    gt_class_mask = gt_classes == class_id

                    # reduce cost matrix to only the relevant gt
                    class_center_dists = class_center_dists[gt_class_mask, :]
                    class_Ng = class_center_dists.shape[0]

                    # create threshold cost matrix
                    threshold_cost_matrix = np.ones((class_Ng, class_Ng)) * 1e6
                    threshold_cost_matrix[np.arange(class_Ng), np.arange(class_Ng)] = (
                        threshold
                    )

                    # create overall cost matrix
                    cost_matrix = np.concatenate(
                        (class_center_dists, threshold_cost_matrix), axis=1
                    )

                    # get optimal assignment
                    _, det_idcs = linear_sum_assignment(cost_matrix)

                    # check if a det is associated to a gt
                    det_is_assoiated = np.isin(np.arange(class_Nt), det_idcs)

                    # create tps and fps
                    class_det_tps = det_is_assoiated.astype(int)
                    class_det_fps = np.logical_not(det_is_assoiated).astype(int)
                    class_det_confidences = class_det_conf

                elif self._match_strategy == "greedy":  # NuScenes style
                    raise NotImplementedError("Greedy matching not yet implemented.")
                else:
                    raise ValueError(
                        f"Match strategy {self._match_strategy} not supported."
                    )

                # append data
                res[class_id.item()]["class_tp"] += np.sum(class_det_tps).item()
                res[class_id.item()]["class_tps"] = np.concatenate(
                    (res[class_id.item()]["class_tps"], class_det_tps)
                )
                res[class_id.item()]["class_fp"] += np.sum(class_det_fps).item()
                res[class_id.item()]["class_fps"] = np.concatenate(
                    (res[class_id.item()]["class_fps"], class_det_fps)
                )
                res[class_id.item()]["class_confidences"] = np.concatenate(
                    (res[class_id.item()]["class_confidences"], class_det_confidences)
                )

        # accumulate TP and FP
        for class_id in range(num_classes):
            class_tps = res[class_id]["class_tps"]
            class_fps = res[class_id]["class_fps"]
            class_Ng = res[class_id]["class_num_gt"]
            class_Nt = class_tps.shape[0]

            class_det_confidences = res[class_id]["class_confidences"]

            (
                class_recall_interp,
                class_precision_interp,
                class_confidence_interp,
                class_ap,
                sum_tp,
                sum_fp,
                sum_fn,
            ) = self._accumulate(class_Ng, class_det_confidences, class_tps, class_fps)

            res[class_id]["class_ap"] = (
                class_ap.item() if isinstance(class_ap, np.generic) else class_ap
            )
            res[class_id]["class_precision"] = class_precision_interp
            res[class_id]["class_recall"] = class_recall_interp
            res[class_id]["class_confidence"] = class_confidence_interp
            res[class_id]["class_tp"] = sum_tp
            res[class_id]["class_fp"] = sum_fp
            res[class_id]["class_fn"] = sum_fn

        return res

    def _accumulate(self, class_Ng, class_det_confidences, class_tps, class_fps):
        class_Nt = class_tps.shape[0]

        if class_Nt == 0 or class_Ng == 0:
            fn = class_Ng
            fp = class_Nt
            tp = 0
            return np.linspace(0, 1, 101), np.zeros(101), np.zeros(101), 0.0, tp, fp, fn

        min_recall = self.config["RECALL_THRESHOLD"]
        min_precision = self.config["PRECISION_THRESHOLD"]

        sort_idcs = np.argsort(class_det_confidences)[::-1]
        class_tps_sorted = class_tps[sort_idcs]
        class_fps_sorted = class_fps[sort_idcs]
        class_conf_sorted = class_det_confidences[sort_idcs]

        class_tp = np.cumsum(class_tps_sorted).astype(float)
        class_fp = np.cumsum(class_fps_sorted).astype(float)

        # calculate precision and recall
        class_precision = class_tp / (class_tp + class_fp)
        class_recall = class_tp / float(class_Ng)

        class_recall_interp = np.linspace(0, 1, 101)
        class_precision_interp = np.interp(
            class_recall_interp, class_recall, class_precision, right=0
        )
        class_confidence_interp = np.interp(
            class_recall_interp, class_recall, class_conf_sorted, right=0
        )

        # calculate AP
        clipped_class_precision = class_precision_interp[round(100 * min_recall) + 1 :]
        clipped_class_precision -= min_precision
        clipped_class_precision[clipped_class_precision < 0] = 0.0

        if self._clipping_mode == "nusc":
            class_ap = np.mean(clipped_class_precision) / (1.0 - min_precision)
        else:
            class_ap = np.mean(clipped_class_precision) + min_precision

        sum_tp = int(class_tp[-1])
        sum_fp = int(class_fp[-1])
        sum_fn = int(class_Ng - class_tp[-1])

        return (
            class_recall_interp,
            class_precision_interp,
            class_confidence_interp,
            class_ap,
            sum_tp,
            sum_fp,
            sum_fn,
        )

    def _prec_re_to_plotly(self, prec, rec):
        """Converts precision and recall to a plotly figure."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.logger.warning(
                "Plotly is not installed. Cannot create precision-recall curve."
            )
            self.visualize = False
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Precision-Recall"))
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )
        return fig

    def combine_batches(self, all_res):
        """Combines the results from all batches."""
        class_ids = all_res[0]["map_raw"][self.config["THRESHOLDS"][0]].keys()
        comb_res = {
            threshold: {
                cls_id: {
                    "tps": np.concatenate(
                        [
                            seq["map_raw"][threshold][cls_id]["class_tps"]
                            for seq in all_res.values()
                        ]
                    ),
                    "fps": np.concatenate(
                        [
                            seq["map_raw"][threshold][cls_id]["class_fps"]
                            for seq in all_res.values()
                        ]
                    ),
                    "confidences": np.concatenate(
                        [
                            seq["map_raw"][threshold][cls_id]["class_confidences"]
                            for seq in all_res.values()
                        ]
                    ),
                    "num_gt": np.sum(
                        [
                            seq["map_raw"][threshold][cls_id]["class_num_gt"]
                            for seq in all_res.values()
                        ]
                    ),
                }
                for cls_id in all_res[0]["map_raw"][threshold].keys()
            }
            for threshold in self.config["THRESHOLDS"]
        }

        for threshold in self.config["THRESHOLDS"]:
            for cls_id in all_res[0]["map_raw"][threshold].keys():
                class_Ng = comb_res[threshold][cls_id]["num_gt"]
                class_tps = comb_res[threshold][cls_id]["tps"]
                class_fps = comb_res[threshold][cls_id]["fps"]
                class_confs = comb_res[threshold][cls_id]["confidences"]

                (
                    class_recall_interp,
                    class_precision_interp,
                    class_confidence_interp,
                    class_ap,
                    sum_tp,
                    sum_fp,
                    sum_fn,
                ) = self._accumulate(class_Ng, class_confs, class_tps, class_fps)

                comb_res[threshold][cls_id]["class_ap"] = class_ap
                comb_res[threshold][cls_id]["class_precision"] = class_precision_interp
                comb_res[threshold][cls_id]["class_recall"] = class_recall_interp
                comb_res[threshold][cls_id][
                    "class_confidence"
                ] = class_confidence_interp
                comb_res[threshold][cls_id]["class_tp"] = sum_tp
                comb_res[threshold][cls_id]["class_fp"] = sum_fp
                comb_res[threshold][cls_id]["class_fn"] = sum_fn

        ret_res = {
            "AP": {
                threshold: {
                    self.classes[cls_id] if self.classes is not None else cls_id: {
                        "AP": (
                            comb_res[threshold][cls_id]["class_ap"].item()
                            if isinstance(
                                comb_res[threshold][cls_id]["class_ap"], np.generic
                            )
                            else comb_res[threshold][cls_id]["class_ap"]
                        ),
                        "pr_curve": (
                            self._prec_re_to_plotly(
                                comb_res[threshold][cls_id]["class_precision"],
                                comb_res[threshold][cls_id]["class_recall"],
                            )
                            if self.visualize
                            else None
                        ),
                    }
                    for cls_id in class_ids
                    if cls_id in self._used_class_ids
                }
                for threshold in self.config["THRESHOLDS"]
            },
            "classwise_mAP": {
                self.classes[cls_id] if self.classes is not None else cls_id: np.mean(
                    [
                        comb_res[threshold][cls_id]["class_ap"]
                        for threshold in self.config["THRESHOLDS"]
                    ]
                ).item()
                for cls_id in class_ids
                if cls_id in self._used_class_ids
            },
            "thresholdwise_mAP": {
                threshold: np.mean(
                    [
                        comb_res[threshold][cls_id]["class_ap"]
                        for cls_id in class_ids
                        if cls_id in self._used_class_ids
                    ]
                ).item()
                for threshold in self.config["THRESHOLDS"]
            },
            "mAP": np.mean(
                [
                    np.mean(
                        [
                            comb_res[threshold][cls_id]["class_ap"]
                            for threshold in self.config["THRESHOLDS"]
                        ]
                    )
                    for cls_id in class_ids
                    if cls_id in self._used_class_ids
                ]
            ).item(),
        }

        # collapse nested dict to flat dict, join keys with "/"
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
