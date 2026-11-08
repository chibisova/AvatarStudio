import torch
from mhr.mhr import MHR


# ==========================================
# 1. Load MHR
# ==========================================

mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

skeleton = mhr_model.character.skeleton


# ==========================================
# 2. Helper functions
# ==========================================

def distance(a, b):
    return torch.linalg.vector_norm(a - b).item()


def pos(skeleton_state, name):
    index = skeleton.joint_index(name)
    return skeleton_state[0, index, :3]


# ==========================================
# 3. Identity parameter sweep
# ==========================================

def mesh_width_at_y(vertices, y, tolerance=1.0, max_x=30.0):
    y_mask = torch.abs(vertices[:, 1] - y) < tolerance
    x_mask = torch.abs(vertices[:, 0]) < max_x

    slice_vertices = vertices[y_mask & x_mask]

    if slice_vertices.shape[0] < 10:
        return None

    x_values = slice_vertices[:, 0]

    x_low = torch.quantile(x_values, 0.05).item()
    x_high = torch.quantile(x_values, 0.95).item()

    return x_high - x_low

identity_values = [-3, -2, -1, 0, 1, 2, 3]

for identity_index in range(45):

    print(f"\n=== I_{identity_index} ===")

    for value in identity_values:

        # --------------------------------------
        # Create MHR parameters
        # --------------------------------------

        identity_coeffs = torch.zeros(1, 45)
        identity_coeffs[0, identity_index] = value

        model_parameters = torch.zeros(1, 204)

        face_expr_coeffs = torch.zeros(1, 72)

        # --------------------------------------
        # Run MHR
        # --------------------------------------

        with torch.no_grad():

            vertices, skeleton_state = mhr_model(
                identity_coeffs,
                model_parameters,
                face_expr_coeffs
            )

        # --------------------------------------
        # Calculate measurements
        # --------------------------------------
        mesh_vertices = vertices[0]

        y_min = mesh_vertices[:, 1].min().item()
        y_max = mesh_vertices[:, 1].max().item()

        height = y_max - y_min
        for percentage in [0.45, 0.50]:

            y = y_min + height * percentage

            width = mesh_width_at_y(
                vertices[0],
                y,
                tolerance=1.0,
                max_x=30.0
            )

            print(
                f"    {percentage * 100:.0f}% width = {width:.2f}"
            )

