import os
import sys
import math
import random
import pytest

try:
    import torch
except Exception:
    pytest.skip("PyTorch is not installed - skipping HiLO model test", allow_module_level=True)

# Ensure `src` is on path so `hilo` package can be imported when running tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hilo.model.hilo import HiLO
from hilo.loss.optimal_match_loss import OptimalMatchLoss
import yaml


def make_toy_batch(batch_size=16, sensors=1, detections=1, feat_dim=6):
    """Generate a toy batch where the first two channels encode the desired center (x,y).

    Shape:
        x: (B, S, N, C)
        mask: (B, S, N)
        target_boxes: (B, Q, D)  # Q will be 1 in our test, D matches box head out dim
    """
    # random centers in a small range
    centers = (torch.rand(batch_size, sensors, detections, 2) - 0.5) * 4.0

    # other arbitrary features
    other = torch.zeros(batch_size, sensors, detections, feat_dim - 2)

    x = torch.cat([centers, other], dim=-1)

    # all detections present
    mask = torch.ones(batch_size, sensors, detections, dtype=torch.bool)

    return x, mask


def build_minimal_cfg(feat_dim=6, d_model=16, num_queries=1, num_classes=2, norm_layer="layer"):
    # Minimal config dictionary matching expectations of HiLO
    cfg = {}
    cfg["no_mma_tf_masks"] = True
    cfg["no_fusion_tf_masks"] = True

    cfg["detection_normalization"] = {
        "enabled": True,
        "center": [0.0] * feat_dim,
        "scale": [1.0] * feat_dim,
        "dim": -1,
    }

    cfg["detection_embedding"] = {"enabled": False, "embeddings": []}

    cfg["detection_input_mlp"] = {
        "in_channels": feat_dim,
        "hidden_channels": [d_model],
        "norm": {"name": norm_layer},
    }

    cfg["multi_modal_attention"] = {
        "num_layers": 1,
        "d_model": d_model,
        "nhead": 1,
        "use_positional_encoding": False,
    }

    cfg["fusion_decoder"] = {
        "num_layers": 1,
        "d_model": d_model,
        "nhead": 1,
        "num_queries": num_queries,
        "use_positional_encoding": False,
    }

    # box head output dimension: choose 6 for this toy test
    # layout used here: [cx, cy, ex, ey, sin(yaw), cos(yaw)]
    box_out = 6
    cfg["box_regression_head"] = {
        "in_channels": d_model,
        "hidden_channels": [d_model],
        "out_channels": box_out,
        "denormalization": {
            "output_indices": list(range(box_out)),
            "norm_indices": list(range(box_out)),
        },
        "norm": {"name": norm_layer}
    }

    cfg["classification_head"] = {"in_channels": d_model,
                                  "hidden_channels": [d_model], "out_channels": num_classes,
                                  "norm": {"name": norm_layer}}

    cfg["max_range"] = 100.0
    cfg["inference_score_threshold"] = 0.0

    return cfg


def test_hilo_learns_centers_cuda_or_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    random.seed(0)

    # Train across multiple independent toy batches to avoid overfitting to a single batch
    B = 16
    S = 1
    N = 1
    C = 6
    Q = 1

    num_batches = 100

    # generate dataset on CPU (moved to device per-batch during training)
    data_batches = [make_toy_batch(batch_size=B, sensors=S, detections=N, feat_dim=C) for _ in range(num_batches)]

    cfg = build_minimal_cfg(feat_dim=C, d_model=16, num_queries=Q, num_classes=2)
    box_out = cfg["box_regression_head"]["out_channels"]

    # suppress UserWarning
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    model = HiLO(cfg).to(device)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # helper to compute average center loss over dataset
    def dataset_center_loss(model, batches):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, mb in batches:
                xb_d = xb.to(device)
                mb_d = mb.to(device)
                out = model(xb_d, mb_d)
                pred = out["centers"]
                targ = torch.zeros(B, Q, box_out, device=device)
                targ[:, 0, 0:2] = xb_d[:, 0, 0, 0:2]
                targ[:, 0, 2:4] = 1.0
                targ[:, 0, 4] = 0.0
                targ[:, 0, 5] = 1.0
                total += loss_fn(pred, targ[:, :, :2]).item()
        return total / len(batches)

    initial_loss = dataset_center_loss(model, data_batches)

    # single-epoch training over dataset (one forward/backward per batch)
    model.train()
    for xb, mb in data_batches:
        xb_d = xb.to(device)
        mb_d = mb.to(device)
        opt.zero_grad()
        out = model(xb_d, mb_d)
        pred = out["centers"]
        targ = torch.zeros(B, Q, box_out, device=device)
        targ[:, 0, 0:2] = xb_d[:, 0, 0, 0:2]
        targ[:, 0, 2:4] = 1.0
        targ[:, 0, 4] = 0.0
        targ[:, 0, 5] = 1.0
        loss = loss_fn(pred, targ[:, :, :2])
        loss.backward()
        opt.step()

    final_loss = dataset_center_loss(model, data_batches)

    assert final_loss < initial_loss, f"Model did not learn across batches: initial {initial_loss}, final {final_loss}"

def test_hilo_loss_with_matching_overfits_single_sample():
    """Use the project's OptimalMatchLoss (with matching) and ensure the model can overfit one sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)
    random.seed(1)

    B = 8
    S = 1
    N = 1
    C = 6
    Q = 1
    M = 1  # number of ground-truth targets per sample

    # create one sample and repeat it to form a batch
    x0, mask0 = make_toy_batch(batch_size=1, sensors=S, detections=N, feat_dim=C)
    x = x0.repeat(B, 1, 1, 1)
    mask = mask0.repeat(B, 1, 1)

    cfg = build_minimal_cfg(feat_dim=C, d_model=16, num_queries=Q, num_classes=2)
    box_out = cfg["box_regression_head"]["out_channels"]

    # build target tensor with format expected by OptimalMatchLoss:
    # [x, y, length, width, vx, vy, heading, class_label]
    D = 8
    target = torch.zeros(B, M, D)
    # centers match input
    target[:, 0, 0:2] = x[:, 0, 0, 0:2]
    target[:, 0, 2:4] = 1.0
    target[:, 0, 4:6] = 0.0
    target[:, 0, 6] = 0.0  # heading
    target[:, 0, 7] = 0  # class label
    target = target.to(device)
    target_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    model = HiLO(cfg).to(device)
    model.train()

    # OptimalMatchLoss configuration (use center and classification costs)
    loss_cfg = {
        "regression_loss": {"type": "l1", "kwargs": {} , "weight": 1.0},
        "classification_loss": {"type": "cross_entropy", "kwargs": {}, "weight": 1.0},
        "cost_matrix": {
            "center_distance_weight": 1.0,
            "size_distance_weight": 0.0,
            "velocity_distance_weight": 0.0,
            "classification_weight": 1.0,
            "orientation_distance_weight": 0.0,
            "invalid_target_cost": 1e6,
        },
    }

    loss_module = OptimalMatchLoss(loss_cfg)
    loss_module = loss_module.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # initial loss
    with torch.no_grad():
        pred = model(x.to(device), mask.to(device))
        losses, costs, asso = loss_module(pred, target, target_mask)
        initial_total = losses["total_loss"].item()

    # train to overfit same batch
    n = 0
    total = initial_total
    loss_hist = [initial_total]
    while total > 1e-2:
        opt.zero_grad()
        pred = model(x.to(device), mask.to(device))
        losses, costs, asso = loss_module(pred, target, target_mask)
        total = losses["total_loss"]
        loss_hist.append(total.item())
        total.backward()
        opt.step()
        n += 1
        if n > 1000:
            break
    
    # final loss and sanity checks
    with torch.no_grad():
        pred = model(x.to(device), mask.to(device))
        losses, costs, asso = loss_module(pred, target, target_mask)
        final_total_train = losses["total_loss"].item()
        pred_centers = pred["centers"]
        mae_train = torch.mean(torch.abs(pred_centers - target[:, : , :2].to(pred_centers.device))).item()
    
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device), mask.to(device))
        losses, costs, asso = loss_module(pred, target, target_mask)
        final_total = losses["total_loss"].item()
        pred_centers = pred["centers"]
        mae = torch.mean(torch.abs(pred_centers - target[:, : , :2].to(pred_centers.device))).item()

    assert final_total < initial_total, f"Matching loss did not decrease: init {initial_total}, final {final_total}"
    assert mae < 1e-2, f"Predicted centers not close enough after overfitting (MAE={mae})"


def _run_matching_overfit(cfg_model, x, mask, target, target_mask, loss_cfg, device, steps=200, lr=5e-2):
    """Helper: train model with given loss_cfg on repeated batch and return diagnostics."""
    model = HiLO(cfg_model).to(device)
    model.train()

    loss_module = OptimalMatchLoss(loss_cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # initial
    with torch.no_grad():
        pred = model(x.to(device), mask.to(device))
        losses_init, costs_init, aso_init = loss_module(pred, target, target_mask)

    # train
    for i in range(steps):
        opt.zero_grad()
        pred = model(x.to(device), mask.to(device))
        losses, costs, aso = loss_module(pred, target, target_mask)
        total = losses["total_loss"]
        total.backward()
        opt.step()

    # final
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device), mask.to(device))
        losses_final, costs_final, aso_final = loss_module(pred, target, target_mask)

    # diagnostics: regression/classification loss values and mean weighted costs
    diagnostics = {
        "init_losses": {k: float(v) for k, v in losses_init.items()},
        "final_losses": {k: float(v) for k, v in losses_final.items()},
        "init_costs_mean": {},
        "final_costs_mean": {},
    }

    if isinstance(costs_init, dict):
        for k, v in costs_init.items():
            try:
                diagnostics["init_costs_mean"][k] = float(v.mean().cpu().numpy())
            except Exception:
                diagnostics["init_costs_mean"][k] = float(v.mean())
        for k, v in costs_final.items():
            try:
                diagnostics["final_costs_mean"][k] = float(v.mean().cpu().numpy())
            except Exception:
                diagnostics["final_costs_mean"][k] = float(v.mean())

    # MAE of centers
    pred_centers = pred["centers"]
    mae = float(torch.mean(torch.abs(pred_centers - target[:, :, :2].to(pred_centers.device))).cpu().numpy())
    diagnostics["mae_centers"] = mae

    return diagnostics


def test_matching_components_and_costs_decrease():
    """Test regression-only, classification-only, and combined matching/loss scenarios."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    random.seed(7)

    B = 8
    S = 1
    N = 1
    C = 6
    Q = 1
    M = 1

    x0, mask0 = make_toy_batch(batch_size=1, sensors=S, detections=N, feat_dim=C)
    x = x0.repeat(B, 1, 1, 1)
    mask = mask0.repeat(B, 1, 1)

    cfg_model = build_minimal_cfg(feat_dim=C, d_model=16, num_queries=Q, num_classes=3)

    # Build ground-truth target (8 dims): x,y,len,w,vx,vy,heading,class
    D = 8
    target = torch.zeros(B, M, D)
    target[:, 0, 0:2] = x[:, 0, 0, 0:2]
    target[:, 0, 2:4] = 1.0
    target[:, 0, 4:6] = 0.0
    target[:, 0, 6] = 0.0
    target[:, 0, 7] = 1  # choose class 1
    target = target.to(device)
    target_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    x = x.to(device)
    mask = mask.to(device)

    # Scenario A: regression-only (center cost + regression loss)
    # Provide both configs; disable classification by setting weight=0.0
    loss_cfg_reg = {
        "regression_loss": {"type": "l2", "kwargs": {}, "weight": 1.0},
        "classification_loss": {"type": "cross_entropy", "kwargs": {}, "weight": 0.0},
        "cost_matrix": {
            "center_distance_weight": 1.0,
            "size_distance_weight": 0.0,
            "velocity_distance_weight": 0.0,
            "classification_weight": 0.0,
            "orientation_distance_weight": 0.0,
            "invalid_target_cost": 1e6,
        },
    }

    diag_reg = _run_matching_overfit(cfg_model, x, mask, target, target_mask, loss_cfg_reg, device, steps=300, lr=1e-2)

    assert diag_reg["final_losses"]["regression_loss"] < diag_reg["init_losses"]["regression_loss"], "Regression loss did not decrease"
    # center weighted cost should decrease
    if "weighted_center_cost" in diag_reg["init_costs_mean"]:
        assert diag_reg["final_costs_mean"]["weighted_center_cost"] <= diag_reg["init_costs_mean"]["weighted_center_cost"] + 1e-6

    # Scenario B: classification-only
    # Provide both configs; disable regression by setting weight=0.0
    loss_cfg_cls = {
        "regression_loss": {"type": "l2", "kwargs": {}, "weight": 0.0},
        "classification_loss": {"type": "cross_entropy", "kwargs": {}, "weight": 1.0},
        "cost_matrix": {
            "center_distance_weight": 0.0,
            "size_distance_weight": 0.0,
            "velocity_distance_weight": 0.0,
            "classification_weight": 1.0,
            "orientation_distance_weight": 0.0,
            "invalid_target_cost": 1e6,
        },
    }

    diag_cls = _run_matching_overfit(cfg_model, x, mask, target, target_mask, loss_cfg_cls, device, steps=300, lr=1e-2)

    assert "classification_loss" in diag_cls["init_losses"] and "classification_loss" in diag_cls["final_losses"], "Classification loss missing"
    assert diag_cls["final_losses"]["classification_loss"] < diag_cls["init_losses"]["classification_loss"], "Classification loss did not decrease"
    if "weighted_classification_cost" in diag_cls["init_costs_mean"]:
        assert diag_cls["final_costs_mean"]["weighted_classification_cost"] <= diag_cls["init_costs_mean"]["weighted_classification_cost"] + 1e-6

    # Scenario C: combined
    loss_cfg_both = {
        "regression_loss": {"type": "l2", "kwargs": {}, "weight": 1.0},
        "classification_loss": {"type": "cross_entropy", "kwargs": {}, "weight": 1.0},
        "cost_matrix": {
            "center_distance_weight": 1.0,
            "size_distance_weight": 0.0,
            "velocity_distance_weight": 0.0,
            "classification_weight": 10.0,
            "orientation_distance_weight": 0.0,
            "invalid_target_cost": 1e6,
        },
    }

    diag_both = _run_matching_overfit(cfg_model, x, mask, target, target_mask, loss_cfg_both, device, steps=300, lr=5e-3)

    assert diag_both["final_losses"]["regression_loss"] < diag_both["init_losses"]["regression_loss"], "Combined: regression loss did not decrease"
    assert diag_both["final_losses"]["classification_loss"] < diag_both["init_losses"]["classification_loss"], "Combined: classification loss did not decrease"
    if "weighted_center_cost" in diag_both["init_costs_mean"]:
        assert diag_both["final_costs_mean"]["weighted_center_cost"] <= diag_both["init_costs_mean"]["weighted_center_cost"] + 1e-6
    if "weighted_classification_cost" in diag_both["init_costs_mean"]:
        assert diag_both["final_costs_mean"]["weighted_classification_cost"] <= diag_both["init_costs_mean"]["weighted_classification_cost"] + 1e-6

def test_user_config_can_learn():
    """Load the user-provided HiLO config YAML, fill placeholders, and check it can learn on toy data."""
    try:
        import yaml as _yaml
    except Exception:
        pytest.skip("PyYAML not installed; skipping config load test")

    cfg_path = os.path.join(ROOT, "src", "hilo", "config", "model", "hilo.yaml")
    if not os.path.exists(cfg_path):
        pytest.skip("Config file not found: hilo.yaml")

    with open(cfg_path, "r") as f:
        raw = f.read()

    # Minimal placeholder substitutions for this test
    subs = {
        "${data.dim}": "6",
        "${data.number_of_classes}": "3",
        "${model.max_range}": "50.0",
        "${model.input_dim}": "6",
        "${data.number_of_sensors}": "1",
        "${model.dropout}": "0.3",
        "${model.multi_modal_attention.d_model}": "64",
        "${mult:${.d_model},2}": "128",
        "${model.fusion_decoder.d_model}": "64",
        "${add:${model.number_of_classes},1}": "4",
    }

    for k, v in subs.items():
        raw = raw.replace(k, v)

    cfg = _yaml.safe_load(raw)

    # Prepare toy data consistent with cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 8
    S = 1
    N = 1
    C = int(cfg.get("input_dim", 6))

    x0, mask0 = make_toy_batch(batch_size=1, sensors=S, detections=N, feat_dim=C)
    x = x0.repeat(B, 1, 1, 1).to(device)
    mask = mask0.repeat(B, 1, 1).to(device)

    # Build GT in expected format for OptimalMatchLoss
    D = 8
    target = torch.zeros(B, 1, D, device=device)
    target[:, 0, 0:2] = x[:, 0, 0, 0:2]
    target[:, 0, 2:4] = 1.0
    target[:, 0, 4:6] = 0.0
    target[:, 0, 6] = 0.0
    target[:, 0, 7] = 1
    target_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

    # Create model from cfg and a matching loss
    model = HiLO(cfg).to(device)
    loss_cfg = {
        "regression_loss": {"type": "l2", "kwargs": {}, "weight": 1.0},
        "classification_loss": {"type": "cross_entropy", "kwargs": {}, "weight": 1.0},
        "cost_matrix": {
            "center_distance_weight": 1.0,
            "size_distance_weight": 0.0,
            "velocity_distance_weight": 0.0,
            "classification_weight": 1.0,
            "orientation_distance_weight": 0.0,
            "invalid_target_cost": 1e6,
        },
    }

    loss_module = OptimalMatchLoss(loss_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        pred = model(x, mask)
        losses_init, costs_init, aso_init = loss_module(pred, target, target_mask)
        init_total = float(losses_init["total_loss"].cpu().numpy())

    # train a bit
    model.train()
    for _ in range(200):
        opt.zero_grad()
        pred = model(x, mask)
        losses, costs, aso = loss_module(pred, target, target_mask)
        total = losses["total_loss"]
        total.backward()
        opt.step()

    with torch.no_grad():
        pred = model(x, mask)
        losses_final, costs_final, aso_final = loss_module(pred, target, target_mask)
        final_total = float(losses_final["total_loss"].cpu().numpy())

    assert final_total < init_total, f"User config model did not learn: init {init_total}, final {final_total}"


