import plotly.graph_objects as go
import numpy as np
import torch
from hilo.utils.visualization.boxes import get_box_corners_closed


def plot_sample(model_output, data, batch_idx=0, score_threshold=0.1, class_names=None, filter_background=True):
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
    pred_boxes = torch.cat([
        model_output["centers"][batch_idx],
        model_output["extents"][batch_idx],
        model_output["velocities"][batch_idx],
        model_output["yaws"][batch_idx,..., None]
    ], dim=-1).detach().cpu().numpy()
    
    pred_scores = torch.gather(
        model_output["class_probs"][batch_idx],
        -1,
        model_output["class_ids"][batch_idx][..., None]
    ).squeeze(-1).detach().cpu().numpy()

    if filter_background:
        pred_mask = (model_output["class_ids"][batch_idx] < model_output["class_probs"].shape[-1] - 1).detach().cpu().numpy()  # Exclude background class
        pred_mask &= pred_scores > score_threshold
    else:
        pred_mask = pred_scores > score_threshold
    
    pred_boxes = pred_boxes[pred_mask]
    pred_scores = pred_scores[pred_mask]
    pred_classes = model_output["class_ids"][batch_idx][pred_mask].detach().cpu().numpy()
    
    for box, score, class_id in zip(pred_boxes, pred_scores, pred_classes):
        if score > score_threshold:  # Threshold for visualization
            x, y, dx, dy, v_x, v_y, heading = box
            if class_names is not None:
                class_name = class_names[int(class_id)]
            
            corners = get_box_corners_closed(x, y, dx, dy, heading)
            fig.add_trace(
                go.Scatter(
                    x=corners[:, 0],
                    y=corners[:, 1],
                    mode='lines',
                    line=dict(color='blue'),
                    name='Predicted Box',
                    hoverinfo='text',
                    hovertext=f'Class: {class_name if class_names is not None else int(class_id)}, Score: {score:.2f}'
                    )
            )

    # Plot ground truth boxes
    gt_boxes = data['gt_data'][batch_idx][data["gt_mask"][batch_idx]].detach().cpu().numpy()
    for box in gt_boxes:
        x, y, dx, dy, v_x, v_y, heading, class_id, p_exist, time = box
        if class_names is not None:
            class_name = class_names[int(class_id)]
        
        corners = get_box_corners_closed(x, y, dx, dy, heading)
        fig.add_trace(
            go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                mode='lines',
                line=dict(color='green'),
                name='Ground Truth Box',
                hoverinfo='text',
                hovertext=f'Class: {class_name if class_names is not None else int(class_id)}, p_exist: {p_exist:.2f}',
                )
            )

    # set figure to span +- 50m in both x and y
    fig.update_layout(
        # title="BEV Visualization",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        xaxis=dict(range=[-50, 50], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-50, 50]),
        # showlegend=True,
    )
    
    return fig