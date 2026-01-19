import plotly.graph_objects as go
import numpy as np
from hilo.utils.visualization.boxes import get_box_corners_closed


def visualize_metrics_data(prep_batch, batch_idx=0, class_names=None):
    """
    Visualizes the packed data used for metric calculation.

    Args:
        prep_batch (dict): The output of MetricCalculator._pack_data.
        batch_idx (int): The index of the sample within the batch to visualize.
        class_names (list, optional): List of class names corresponding to class IDs.

    Returns:
        go.Figure: Plotly figure.
    """

    fig = go.Figure()

    # Ground Truths
    gt_boxes = prep_batch["gt_boxes"][batch_idx]
    gt_classes = prep_batch["gt_classes"][batch_idx]

    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        cls_id = int(gt_classes[i])

        # x, y, dx, dy, vx, vy, heading
        corners = get_box_corners_closed(box[0], box[1], box[2], box[3], box[6])

        cls_str = str(cls_id) if class_names is None else class_names[cls_id]

        fig.add_trace(
            go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                mode="lines",
                line=dict(color="green"),
                name=f"GT {cls_str}",
                text=f"GT: {cls_str}",
                showlegend=False,
            )
        )

        # Add a marker for the heading
        front_x = box[0] + (box[2] / 2) * np.cos(box[6])
        front_y = box[1] + (box[2] / 2) * np.sin(box[6])
        fig.add_trace(
            go.Scatter(
                x=[box[0], front_x],
                y=[box[1], front_y],
                mode="lines",
                line=dict(color="green", width=2),
                showlegend=False,
            )
        )

    # Detections
    det_boxes = prep_batch["detector_boxes"][batch_idx]
    det_classes = prep_batch["detector_classes"][batch_idx]
    det_scores = prep_batch["detector_confidences"][batch_idx]

    for i in range(len(det_boxes)):
        box = det_boxes[i]
        cls_id = int(det_classes[i])
        score = det_scores[i]

        corners = get_box_corners_closed(box[0], box[1], box[2], box[3], box[6])

        cls_str = str(cls_id) if class_names is None else class_names[cls_id]

        fig.add_trace(
            go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                mode="lines",
                line=dict(color="blue"),
                name=f"Det {cls_str}",
                text=f"Det: {cls_str} ({score:.2f})",
                showlegend=False,
                hoverinfo="text",
            )
        )

        # Add a marker for the heading
        front_x = box[0] + (box[2] / 2) * np.cos(box[6])
        front_y = box[1] + (box[2] / 2) * np.sin(box[6])
        fig.add_trace(
            go.Scatter(
                x=[box[0], front_x],
                y=[box[1], front_y],
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"Sample {batch_idx} Visualization",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True,
    )

    # Add dummy traces for legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="green"),
            name="Ground Truth",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color="blue"), name="Prediction"
        )
    )

    return fig
