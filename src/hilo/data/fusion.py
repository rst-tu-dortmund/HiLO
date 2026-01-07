import pickle
import os
import torch
from torch.utils.data import Dataset

import logging
from typing import List, Tuple, Dict, Any
import numpy as np

GT_FIELDS = (
    "x",
    "y",
    "length",
    "width",
    "v_x",
    "v_y",
    "heading",
    "class_id",
    "p_exist",
    "time",
)


def _sensor_index_map(sensors: List[Tuple[str, str]]) -> Dict[str, int]:
    # sensors: [(sensor_tag, type_name), ...]
    return {tag: idx for idx, (tag, _type) in enumerate(sensors)}


def map_class_labels(class_ids: np.ndarray) -> np.ndarray:
    """
    Map raw class IDs to contiguous class labels starting from 0.
    Example mapping (customize as needed):
      9 -> 0 (truck)
      7 -> 1 (car)
      10 -> 2 (motorcycle)
      11 -> 3 (bicycle)
      12 -> 4 (pedestrian)
      others -> ? (unknown)
    """
    mapped = np.zeros_like(class_ids, dtype=np.float32)

    # class ids come as floats, so use np.isclose for comparison
    mapped[np.isclose(class_ids, 9)] = 0.0
    mapped[np.isclose(class_ids, 7)] = 1.0
    mapped[np.isclose(class_ids, 10)] = 2.0
    mapped[np.isclose(class_ids, 11)] = 3.0
    mapped[np.isclose(class_ids, 12)] = 4.0

    # check if there are any unknown classes
    known_mask = (
        np.isclose(class_ids, 9)
        | np.isclose(class_ids, 7)
        | np.isclose(class_ids, 10)
        | np.isclose(class_ids, 11)
        | np.isclose(class_ids, 12)
    )
    mapped[~known_mask] = -1.0  # unknown class

    if np.any(~known_mask):
        logging.getLogger(__name__).debug(
            f"Unknown class IDs found and mapped to -1: {np.unique(class_ids[~known_mask])}"
        )

    return mapped


def _safe_col(arr: np.ndarray, name: str, count: int) -> np.ndarray:
    # Returns a float32 view if field exists, otherwise zeros
    names = arr.dtype.names
    if names and name in names:
        v = arr[name]
        if v.dtype != np.float32:
            v = v.astype(np.float32, copy=False)
        return v
    return np.zeros((count,), dtype=np.float32)


def _convert_structured_to_feat_array(arr: np.ndarray, sensor_id: int) -> np.ndarray:
    """
    Map structured dtype array to (num_objs, 18) in the specified order:
      0 x, 1 y, 2 length, 3 width,
      4 v_x, 5 v_y, 6 yaw(heading),
      7 class_id, 8 p_exist,
      9 x_std, 10 y_std, 11 length_std, 12 width_std,
      13 v_x_std_dev, 14 v_y_std_dev, 15 yaw_std,
      16 sensor_id, 17 time
    """
    num = len(arr)
    out = np.zeros((num, 18), dtype=np.float32)
    if num == 0:
        return out

    # Required numeric columns
    out[:, 0] = _safe_col(arr, "x", num)
    out[:, 1] = _safe_col(arr, "y", num)
    out[:, 2] = _safe_col(arr, "length", num)
    out[:, 3] = _safe_col(arr, "width", num)
    out[:, 4] = _safe_col(arr, "v_x", num)
    out[:, 5] = _safe_col(arr, "v_y", num)
    out[:, 6] = _safe_col(arr, "heading", num)  # yaw
    out[:, 7] = map_class_labels(_safe_col(arr, "class_id", num))  # cast to float32
    out[:, 8] = _safe_col(arr, "p_exist", num)
    out[:, 9] = _safe_col(arr, "x_std_dev", num)
    out[:, 10] = _safe_col(arr, "y_std_dev", num)
    out[:, 11] = _safe_col(arr, "length_std_dev", num)
    out[:, 12] = _safe_col(arr, "width_std_dev", num)
    out[:, 13] = _safe_col(arr, "v_x_std_dev", num)
    out[:, 14] = _safe_col(arr, "v_y_std_dev", num)
    out[:, 15] = _safe_col(arr, "heading_std_dev", num)  # yaw_std
    out[:, 16] = np.float32(sensor_id)  # sensor id (broadcast)
    out[:, 17] = _safe_col(arr, "time", num)  # timestamp
    return out


def format_sample_to_numpy(
    sample: Dict[str, Any],
    sensors: List[Tuple[str, str]],
    logger: logging.Logger = None,
):
    """
    Inputs:
      sample['data_in']: list where each entry looks like:
        [structured_np_array, (sensor_tag, type_name)]
        e.g., [array([...], dtype=[('x','<f4'), ...]), ('RADAR_RL0','ASW_RdrObjDataAppl_t')]
      sensors: canonical sensor list with tags used for indexing

    Returns:
      data: np.ndarray of shape (S, N, 18)
      valid_mask: np.ndarray bool of shape (S, N), True indicates valid (non-padded)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    tag_to_idx = _sensor_index_map(sensors)
    S = len(sensors)

    # Collect included sensor arrays (skip unknown sensors but log them)
    included = []
    for item in sample.get("data_in", []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        arr = item[0]
        sensor_info = item[1]
        tag = sensor_info[0] if isinstance(sensor_info, (list, tuple)) else sensor_info
        s_idx = tag_to_idx.get(tag, None)
        if s_idx is None:
            logger.debug(f"Unknown sensor tag in sample: {tag}")
            continue
        included.append((s_idx, tag, arr))

    # Determine N = max objects across included sensors
    N = max((len(arr) for _, __, arr in included), default=1)

    data = np.zeros((S, N, 18), dtype=np.float32)
    valid_mask = np.zeros((S, N), dtype=bool)  # True means valid

    # Fill per-sensor data
    for s_idx, _tag, arr in included:
        converted = _convert_structured_to_feat_array(arr, s_idx)
        k = min(len(converted), N)
        if k:
            data[s_idx, :k, :] = converted[:k]
            valid_mask[s_idx, :k] = True

    return data, valid_mask


def _convert_gt_to_array(
    arr: np.ndarray, fields: Tuple[str, ...] = GT_FIELDS
) -> np.ndarray:
    """Convert a structured numpy GT array to (num, len(fields)) float32 in the given order.
    Missing fields are filled with zeros.
    """
    num = len(arr)
    out = np.zeros((num, len(fields)), dtype=np.float32)
    if num == 0:
        return out
    # Vectorized fill per field
    for i, name in enumerate(fields):
        out[:, i] = (
            _safe_col(arr, name, num)
            if name != "class_id"
            else map_class_labels(_safe_col(arr, name, num))
        )
    return out


def format_gt_to_numpy(
    sample: Dict[str, Any],
    logger: logging.Logger = None,
    fields: Tuple[str, ...] = GT_FIELDS,
):
    """
    Assume sample['data_gt'] is a structured numpy array with named fields.
    Returns:
      gt_data: (M, F) float32 where F = len(fields) and M is number of GT objects.
    No padding or mask is produced since GT contains only true objects.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    arr: np.ndarray = sample.get("data_gt", None)
    if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
        F = len(fields)
        return np.zeros((0, F), dtype=np.float32)

    feats = _convert_gt_to_array(arr, fields)
    M, F = feats.shape
    if M == 0:
        return np.zeros((0, F), dtype=np.float32)
    return feats


class FusionFilteredDataset(Dataset):
    def __init__(self, cfg, split="train"):
        super(FusionFilteredDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.downsample_factor = int(cfg.get("downsample_factor", 1))
        self.root = os.path.normpath(cfg["data_root"])
        self.preprocessed_root = os.path.join(self.root, "preprocessed")
        self.preprocess = cfg.get("preprocess", False)
        self.sensors = [
            ("RADAR_RR0", "ASW_RdrObjDataAppl_t"),
            ("RADAR_RL0", "ASW_RdrObjDataAppl_t"),
            ("RADAR_FR0", "ASW_RdrObjDataAppl_t"),
            ("RADAR_FL0", "ASW_RdrObjDataAppl_t"),
            ("CAMERA_FC0", "ASW_VidObjData_t"),
        ]

        with open(os.path.join(cfg["data_path"], f"{split}.p"), "rb") as f:
            rel_data_paths = pickle.load(f)

        self.rel_data_paths = rel_data_paths[:: self.downsample_factor]
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return super().__str__() + f"(split={self.split}, num_samples={len(self)})"

    def format_sample(self, sample):
        data, mask = format_sample_to_numpy(sample, self.sensors, self.logger)
        # GT is returned as (M, 10) without padding/mask
        gt_data = format_gt_to_numpy(sample, logger=self.logger)

        # timestamp alignment: set data[..., -1] to time difference to GT first object's time in ms
        if gt_data.shape[0] > 0:
            data[..., -1] = gt_data[0, -1] - data[..., -1]

        # create formatted sample for the neural network
        data_t = torch.from_numpy(data).to(torch.float32)
        mask_t = torch.from_numpy(mask).to(torch.bool)
        gt_data_t = torch.from_numpy(gt_data).to(torch.float32)

        formatted_sample = {
            "data": data_t,  # shape: (S, N, 18)
            "mask": mask_t,  # shape: (S, N) - True means valid
            "gt_data": gt_data_t,  # shape: (M, 10)
        }

        return formatted_sample

    def load_sample(self, idx):
        if self.preprocess:
            preprocessed_data_path = os.path.join(
                self.preprocessed_root, self.rel_data_paths[idx].replace("\\", "/")
            )
            if os.path.exists(preprocessed_data_path):
                with open(preprocessed_data_path, "rb") as f:
                    sample = pickle.load(f)
                return sample

        data_path = os.path.join(self.root, self.rel_data_paths[idx].replace("\\", "/"))
        with open(data_path, "rb") as f:
            sample = pickle.load(f)

        sample = self.format_sample(sample)

        if self.preprocess:
            os.makedirs(
                os.path.dirname(preprocessed_data_path),
                exist_ok=True,
            )
            with open(preprocessed_data_path, "wb") as f:
                pickle.dump(sample, f)

        return sample

    def __len__(self):
        return len(self.rel_data_paths)

    def __getitem__(self, idx):
        sample = self.load_sample(idx)
        return sample

    def collate_fn(self, batch):
        """
        Batch a list of formatted samples (torch Tensors) by padding to the max sizes in the batch.
        Returns torch tensors:
          - data: (B, S, N_max, 18)
          - mask: (B, S, N_max) boolean, True indicates valid (non-padded)
          - gt_data: (B, M_max, 10)
          - gt_mask: (B, M_max) boolean, True indicates valid (non-padded)
        """
        B = len(batch)
        # All samples share same S and feature dims
        S = batch[0]["data"].shape[0]
        C = batch[0]["data"].shape[2]
        device = batch[0]["data"].device
        data_dtype = batch[0]["data"].dtype
        mask_dtype = torch.bool

        # Feature count for GT
        F = (
            batch[0]["gt_data"].shape[1]
            if batch[0]["gt_data"].ndim == 2 and batch[0]["gt_data"].shape[0] > 0
            else len(GT_FIELDS)
        )
        gt_dtype = batch[0]["gt_data"].dtype

        # Max across the batch
        N_max = max(s["data"].shape[1] for s in batch)
        M_max = max((s["gt_data"].shape[0] for s in batch), default=0)

        # Allocate batched tensors
        data = torch.zeros((B, S, N_max, C), dtype=data_dtype, device=device)
        mask = torch.zeros(
            (B, S, N_max), dtype=mask_dtype, device=device
        )  # True = valid
        gt_data = (
            torch.zeros((B, M_max, F), dtype=gt_dtype, device=device)
            if M_max > 0
            else torch.zeros((B, 0, F), dtype=gt_dtype, device=device)
        )
        gt_mask = (
            torch.zeros((B, M_max), dtype=mask_dtype, device=device)
            if M_max > 0
            else torch.zeros((B, 0), dtype=mask_dtype, device=device)
        )  # True = valid

        for i, sample in enumerate(batch):
            d = sample["data"]  # (S, N_i, C)
            m = sample["mask"]  # (S, N_i), True = valid
            N_i = d.shape[1]

            data[i, :, :N_i, :] = d
            mask[i, :, :N_i] = m

            g = sample["gt_data"]  # (M_i, F)
            M_i = g.shape[0]
            if M_max > 0 and M_i > 0:
                gt_data[i, :M_i, :] = g
                gt_mask[i, :M_i] = True

        return {
            "data": data,
            "mask": mask,
            "gt_data": gt_data,
            "gt_mask": gt_mask,
        }
