import torch
import trimesh
import csv

from mhr.mhr import MHR


# ==========================================
# 1. Load MHR
# ==========================================

mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

faces = mhr_model.character.mesh.faces


# ==========================================
# 2. Generate mesh function
# ==========================================

def generate_mesh(identity_coeffs):

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
# 3. Generate baseline
# ==========================================

baseline_coeffs = torch.zeros(1, 45)

baseline_vertices = generate_mesh(
    baseline_coeffs
)


# ==========================================
# 4. Analyze all parameters
# ==========================================

results = []

for parameter_index in range(45):

    print(f"Analyzing parameter {parameter_index}...")

    # Positive variation
    positive_coeffs = torch.zeros(1, 45)
    positive_coeffs[0, parameter_index] = 1.0

    positive_vertices = generate_mesh(
        positive_coeffs
    )

    # Negative variation
    negative_coeffs = torch.zeros(1, 45)
    negative_coeffs[0, parameter_index] = -1.0

    negative_vertices = generate_mesh(
        negative_coeffs
    )

    # Calculate vertex displacement
    positive_displacement = (
        (positive_vertices - baseline_vertices) ** 2
    ).sum(axis=1) ** 0.5

    negative_displacement = (
        (negative_vertices - baseline_vertices) ** 2
    ).sum(axis=1) ** 0.5

    positive_mean = positive_displacement.mean()
    negative_mean = negative_displacement.mean()

    positive_max = positive_displacement.max()
    negative_max = negative_displacement.max()

    results.append({
        "parameter": parameter_index,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_max": positive_max,
        "negative_max": negative_max
    })


# ==========================================
# 5. Save results to CSV
# ==========================================

with open(
    "identity_analysis.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "parameter",
            "positive_mean",
            "negative_mean",
            "positive_max",
            "negative_max"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("Analysis complete!")
print("Saved results to identity_analysis.csv")
