import plotly.graph_objects as go
import numpy as np
import torch
from hilo.utils.visualization.boxes import get_box_corners_closed


def plot_sample(
    model_output,
    data,
    batch_idx=0,
    score_threshold=0.1,
    class_names=None,
    filter_background=True,
):
    """
    Plots a BEV visualization of the model output and ground truth data for a given batch index.

    Args:
        model_output (dict): The output from the model containing predicted boxes and scores.
        data (dict): The input data containing ground truth boxes.
        batch_idx (int): The index of the batch to visualize.

    Returns:
        fig (go.Figure): A Plotly figure object containing the BEV visualization.
    """
    if class_names is not None:
        class_names.append("no object")  # Add background class name

    fig = go.Figure()

    # Plot predicted boxes
    pred_boxes = (
        torch.cat(
            [
                model_output["centers"][batch_idx],
                model_output["extents"][batch_idx],
                model_output["velocities"][batch_idx],
                model_output["yaws"][batch_idx, ..., None],
            ],
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    pred_scores = (
        torch.gather(
            model_output["class_probs"][batch_idx],
            -1,
            model_output["class_ids"][batch_idx][..., None],
        )
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
    )

    if filter_background:
        pred_mask = (
            (
                model_output["class_ids"][batch_idx]
                < model_output["class_probs"].shape[-1] - 1
            )
            .detach()
            .cpu()
            .numpy()
        )  # Exclude background class
        pred_mask &= pred_scores > score_threshold
    else:
        pred_mask = pred_scores > score_threshold

    pred_boxes = pred_boxes[pred_mask]
    pred_scores = pred_scores[pred_mask]
    pred_classes = (
        model_output["class_ids"][batch_idx][pred_mask].detach().cpu().numpy()
    )
    pred_objs = np.concatenate(
        (pred_boxes, pred_classes[..., None], pred_scores[..., None]),
        axis=-1,
    )

    pred_label = "Predicted Box"
    render_boxes_with_scores(
        score_threshold, class_names, fig, pred_objs, pred_label, "blue"
    )

    # Plot ground truth boxes
    gt_boxes = (
        data["gt_data"][batch_idx][data["gt_mask"][batch_idx]].detach().cpu().numpy()
    )
    gt_label = "Ground Truth Box"
    render_boxes_with_scores(0.0, class_names, fig, gt_boxes, gt_label, "green")

    # set scale ratio to be equal and set a reasonable figure size
    fig.update_layout(
        # title="BEV Visualization",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=800,
        height=800,
    )

    return fig


def render_boxes_with_scores(
    score_threshold,
    class_names,
    fig,
    objs,
    obj_label,
    color,
    additional_hover_texts=None,
):
    for k, box in enumerate(objs):
        x, y, dx, dy, v_x, v_y, heading, class_id, score = box[..., :9]
        if score > score_threshold:  # Threshold for visualization
            if class_names is not None:
                class_name = class_names[int(class_id)]

            corners = get_box_corners_closed(x, y, dx, dy, heading)
            hovertext = f"Class: {class_name if class_names is not None else int(class_id)}<br>Score: {score:.2f}"
            if additional_hover_texts is not None:
                for key, text in additional_hover_texts[k].items():
                    hovertext += f"<br>{key}: {text}"

            fig.add_trace(
                go.Scatter(
                    x=corners[:, 0],
                    y=corners[:, 1],
                    mode="lines",
                    line=dict(color=color),
                    name=obj_label,
                    hoverinfo="text",
                    hovertext=hovertext,
                )
            )


def plot_association_results(
    association_results,
    batch_idx=0,
    class_names=None,
):
    """
    Plots a BEV visualization of the association results between predicted and ground truth boxes for a given batch index.

    Args:
        association_results (dict): The association results containing matched predicted and ground truth boxes.
        data (dict): The input data containing ground truth boxes.
        batch_idx (int): The index of the batch to visualize.
        class_names (list): The list of class names for visualization.
    """
    fig = go.Figure()

    if class_names is not None:
        class_names.append("no object")  # Add background class name

    matched_batch_mask = (
        association_results[True]["prediction_indices"][..., 0] == batch_idx
    )
    matched_preds = association_results[True]["prediction"]

    # Plot matched predicted boxes
    matched_pred_boxes = (
        torch.cat(
            [
                matched_preds["centers"][matched_batch_mask],
                matched_preds["extents"][matched_batch_mask],
                matched_preds["velocities"][matched_batch_mask],
                matched_preds["yaws"][matched_batch_mask, ..., None],
            ],
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    matched_pred_scores = (
        torch.gather(
            matched_preds["class_probs"][matched_batch_mask],
            -1,
            matched_preds["class_ids"][matched_batch_mask][..., None],
        )
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
    )

    matched_pred_classes = (
        matched_preds["class_ids"][matched_batch_mask].detach().cpu().numpy()
    )

    matched_pred_objs = np.concatenate(
        (
            matched_pred_boxes,
            matched_pred_classes[..., None],
            matched_pred_scores[..., None],
        ),
        axis=-1,
    )

    additional_hover_texts = [
        {
            "match index": i,
        }
        for i in range(matched_pred_objs.shape[0])
    ]

    render_boxes_with_scores(
        0.0,
        class_names,
        fig,
        matched_pred_objs,
        "Matched Predicted Box",
        "blue",
        additional_hover_texts,
    )

    # Plot matched ground truth boxes
    matched_gts = association_results[True]["target"]
    matched_gt_boxes = matched_gts[matched_batch_mask].detach().cpu().numpy()

    render_boxes_with_scores(
        0.0,
        class_names,
        fig,
        matched_gt_boxes,
        "Matched Ground Truth Box",
        "green",
        additional_hover_texts,
    )

    # Plot unmatched predicted boxes
    unmatched_batch_mask = (
        association_results[False]["prediction_indices"][..., 0] == batch_idx
    )
    unmatched_preds = association_results[False]["prediction"]

    unmatched_pred_boxes = (
        torch.cat(
            [
                unmatched_preds["centers"][unmatched_batch_mask],
                unmatched_preds["extents"][unmatched_batch_mask],
                unmatched_preds["velocities"][unmatched_batch_mask],
                unmatched_preds["yaws"][unmatched_batch_mask, ..., None],
            ],
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    unmatched_pred_scores = (
        torch.gather(
            unmatched_preds["class_probs"][unmatched_batch_mask],
            -1,
            unmatched_preds["class_ids"][unmatched_batch_mask][..., None],
        )
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
    )

    unmatched_pred_classes = (
        unmatched_preds["class_ids"][unmatched_batch_mask].detach().cpu().numpy()
    )

    unmatched_pred_objs = np.concatenate(
        (
            unmatched_pred_boxes,
            unmatched_pred_classes[..., None],
            unmatched_pred_scores[..., None],
        ),
        axis=-1,
    )

    render_boxes_with_scores(
        0.0,
        class_names,
        fig,
        unmatched_pred_objs,
        "Unmatched Predicted Box",
        "red",
    )

    # set aspect ratio to be equal and set a reasonable figure size
    fig.update_layout(
        # title="BEV Association Visualization",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=800,
        height=800,
    )

    return fig
