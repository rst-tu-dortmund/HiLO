import numpy as np


def get_box_corners_closed(x, y, dx, dy, heading):
    """
    Computes the corners of a 2D bounding box in BEV given its center, dimensions, and heading.

    Args:
        x (float): Center x-coordinate.
        y (float): Center y-coordinate.
        dx (float): Length of the box along the x-axis.
        dy (float): Length of the box along the y-axis.
        heading (float): Rotation angle around the z-axis in radians.

    Returns:
        corners (np.ndarray): An array of shape (4, 2) representing the corners of the box in BEV.
    """
    # Define the corners relative to the center
    corners = np.array(
        [
            [dx / 2, dy / 2],
            [dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2, dy / 2],
            [dx / 2, dy / 2],
        ]
    )

    # Rotation matrix
    rotation_matrix = np.array(
        [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
    )

    # Rotate and translate corners
    rotated_corners = corners @ rotation_matrix.T
    translated_corners = rotated_corners + np.array([x, y])

    return translated_corners
