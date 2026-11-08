import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mhr.mhr import MHR


# ==========================================
# Configuration
# ==========================================

OUTPUT_DIR = "identity_visualizations_multiview"

PARAMETER_COUNT = 45
VALUE = 1.0

FIGURE_SIZE = (15, 22)
DPI = 150

ELEVATION = 10

# Coordinate system after conversion:
# X = horizontal
# Y = depth
# Z = vertical

VIEWS = {
    "Front": 90,
    "Side": 0,
    "Back": -90
}


# ==========================================
# Load MHR
# ==========================================

print("Loading MHR model...")

mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

faces = np.asarray(
    mhr_model.character.mesh.faces
)

print("Model loaded!")


# ==========================================
# Coordinate conversion
# ==========================================

def to_plot_coordinates(vertices):

    return vertices[:, [0, 2, 1]]


# ==========================================
# Generate vertices
# ==========================================

def generate_vertices(identity_coeffs):

    model_parameters = torch.zeros(1, 204)

    face_expr_coeffs = torch.zeros(1, 72)

    with torch.no_grad():

        vertices, skeleton_state = mhr_model(
            identity_coeffs,
            model_parameters,
            face_expr_coeffs
        )

    return vertices[0].numpy()


# ==========================================
# Output directory
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# Generate baseline
# ==========================================

print("Generating baseline...")

baseline_coeffs = torch.zeros(1, 45)

baseline_vertices = generate_vertices(
    baseline_coeffs
)


# ==========================================
# Render mesh
# ==========================================

def render_mesh(
    ax,
    vertices,
    title,
    axis_limits,
    aspect_ratio,
    azimuth,
    face_values=None,
    color_norm=None
):

    vertices = to_plot_coordinates(
        vertices
    )

    triangle_vertices = vertices[faces]

    # --------------------------------------
    # Normal mesh
    # --------------------------------------

    if face_values is None:

        mesh_collection = Poly3DCollection(
            triangle_vertices,
            facecolors="lightgray",
            edgecolors="none"
        )

    # --------------------------------------
    # Heatmap mesh
    # --------------------------------------

    else:

        triangle_values = face_values[faces]

        face_displacement = triangle_values.mean(
            axis=1
        )

        mesh_collection = Poly3DCollection(
            triangle_vertices,
            cmap="turbo",
            norm=color_norm,
            edgecolors="none"
        )

        mesh_collection.set_array(
            face_displacement
        )

    ax.add_collection3d(
        mesh_collection
    )

    # --------------------------------------
    # Camera and limits
    # --------------------------------------

    ax.set_xlim(
        *axis_limits[0]
    )

    ax.set_ylim(
        *axis_limits[1]
    )

    ax.set_zlim(
        *axis_limits[2]
    )

    ax.set_box_aspect(
        aspect_ratio
    )

    ax.view_init(
        elev=ELEVATION,
        azim=azimuth
    )

    ax.set_proj_type("ortho")

    ax.set_title(
        title,
        fontsize=11
    )

    ax.set_axis_off()

    return mesh_collection


# ==========================================
# Process all parameters
# ==========================================

for parameter_index in range(
    PARAMETER_COUNT
):

    print(
        f"Processing parameter "
        f"{parameter_index + 1}/"
        f"{PARAMETER_COUNT}"
    )

    # --------------------------------------
    # Generate negative mesh
    # --------------------------------------

    negative_coeffs = torch.zeros(
        1, 45
    )

    negative_coeffs[
        0, parameter_index
    ] = -VALUE

    negative_vertices = generate_vertices(
        negative_coeffs
    )

    # --------------------------------------
    # Generate positive mesh
    # --------------------------------------

    positive_coeffs = torch.zeros(
        1, 45
    )

    positive_coeffs[
        0, parameter_index
    ] = VALUE

    positive_vertices = generate_vertices(
        positive_coeffs
    )

    # --------------------------------------
    # Calculate displacement
    # --------------------------------------

    negative_displacement = np.linalg.norm(
        negative_vertices - baseline_vertices,
        axis=1
    )

    positive_displacement = np.linalg.norm(
        positive_vertices - baseline_vertices,
        axis=1
    )

    max_displacement = max(
        negative_displacement.max(),
        positive_displacement.max(),
        1e-8
    )

    color_norm = Normalize(
        vmin=0.0,
        vmax=max_displacement
    )

    # --------------------------------------
    # Calculate plot limits
    # --------------------------------------

    all_vertices = np.vstack([
        baseline_vertices,
        negative_vertices,
        positive_vertices
    ])

    all_plot_vertices = to_plot_coordinates(
        all_vertices
    )

    minimum = all_plot_vertices.min(
        axis=0
    )

    maximum = all_plot_vertices.max(
        axis=0
    )

    margin = 0.1

    axis_limits = [
        (
            minimum[0] - margin,
            maximum[0] + margin
        ),
        (
            minimum[1] - margin,
            maximum[1] + margin
        ),
        (
            minimum[2] - margin,
            maximum[2] + margin
        )
    ]

    aspect_ratio = maximum - minimum

    # ======================================
    # Create figure
    # ======================================

    fig = plt.figure(
        figsize=FIGURE_SIZE,
        dpi=DPI,
        constrained_layout=True
    )

    fig.suptitle(
        f"Identity Parameter {parameter_index}",
        fontsize=20
    )

    grid = fig.add_gridspec(
        6,
        4,
        width_ratios=[
            1,
            1,
            1,
            0.08
        ]
    )

    # ======================================
    # Rows 0–2: Normal views
    # ======================================

    for view_row, (view_name, azimuth) in enumerate(
        VIEWS.items()
    ):

        # Negative
        ax_negative = fig.add_subplot(
            grid[view_row, 0],
            projection="3d"
        )

        render_mesh(
            ax_negative,
            negative_vertices,
            f"{view_name} | -{VALUE}",
            axis_limits,
            aspect_ratio,
            azimuth
        )

        # Neutral
        ax_neutral = fig.add_subplot(
            grid[view_row, 1],
            projection="3d"
        )

        render_mesh(
            ax_neutral,
            baseline_vertices,
            f"{view_name} | 0.0",
            axis_limits,
            aspect_ratio,
            azimuth
        )

        # Positive
        ax_positive = fig.add_subplot(
            grid[view_row, 2],
            projection="3d"
        )

        render_mesh(
            ax_positive,
            positive_vertices,
            f"{view_name} | +{VALUE}",
            axis_limits,
            aspect_ratio,
            azimuth
        )

    # ======================================
    # Rows 3–5: Heatmaps
    # ======================================

    for view_row, (view_name, azimuth) in enumerate(
        VIEWS.items(),
        start=3
    ):

        # Negative heatmap
        ax_negative_heatmap = fig.add_subplot(
            grid[view_row, 0],
            projection="3d"
        )

        render_mesh(
            ax_negative_heatmap,
            negative_vertices,
            f"{view_name} | Negative displacement",
            axis_limits,
            aspect_ratio,
            azimuth,
            face_values=negative_displacement,
            color_norm=color_norm
        )

        # Empty center cell
        ax_empty = fig.add_subplot(
            grid[view_row, 1]
        )

        ax_empty.axis("off")

        # Positive heatmap
        ax_positive_heatmap = fig.add_subplot(
            grid[view_row, 2],
            projection="3d"
        )

        render_mesh(
            ax_positive_heatmap,
            positive_vertices,
            f"{view_name} | Positive displacement",
            axis_limits,
            aspect_ratio,
            azimuth,
            face_values=positive_displacement,
            color_norm=color_norm
        )

    # ======================================
    # Shared colorbar
    # ======================================

    scalar_map = plt.cm.ScalarMappable(
        norm=color_norm,
        cmap="turbo"
    )

    scalar_map.set_array([])

    ax_colorbar = fig.add_subplot(
        grid[:, 3]
    )

    fig.colorbar(
        scalar_map,
        cax=ax_colorbar,
        label="Vertex displacement"
    )

    # ======================================
    # Save image
    # ======================================

    filename = os.path.join(
        OUTPUT_DIR,                 
        f"identity_{parameter_index:02d}.png"
    )

    plt.savefig(
        filename,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {filename}"
    )


print("===================================")
print("All visualizations completed!")
print(f"Output directory: {OUTPUT_DIR}")
print("===================================")
