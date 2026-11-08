import torch
from mhr.mhr import MHR


# ==================================================
# 1. Load MHR
# ==================================================

mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

blend_shape = mhr_model.character.blend_shape


# ==================================================
# 2. Get MHR blendshape data
# ==================================================

base_vertices = torch.tensor(
    blend_shape.base_shape,
    dtype=torch.float32
)

identity_basis = torch.tensor(
    blend_shape.shape_vectors[:45],
    dtype=torch.float32
)

print("Base vertices:", base_vertices.shape)
print("Identity basis:", identity_basis.shape)


# ==================================================
# 3. Create random identity coefficients
# ==================================================

torch.manual_seed(42)

identity_coeffs = 0.8 * torch.randn(1, 45)

print()
print("Identity coefficients:")
print(identity_coeffs)


# ==================================================
# 4. MHR forward pass
# ==================================================

model_parameters = torch.zeros(1, 204)
face_expr_coeffs = torch.zeros(1, 72)

with torch.no_grad():

    mhr_vertices, _ = mhr_model(
        identity_coeffs,
        model_parameters,
        face_expr_coeffs
    )

mhr_vertices = mhr_vertices[0]


# ==================================================
# 5. Reconstruct directly from blendshapes
# ==================================================

coeffs = identity_coeffs[0]

reconstructed_vertices = (
    base_vertices
    + torch.sum(
        coeffs[:, None, None] * identity_basis,
        dim=0
    )
)


# ==================================================
# 6. Compare
# ==================================================

difference = torch.abs(
    mhr_vertices - reconstructed_vertices
)

print()
print("==========================================")
print("Direct blendshape reconstruction test")
print("==========================================")

print("MHR vertices:          ", mhr_vertices.shape)
print("Reconstructed vertices:", reconstructed_vertices.shape)

print()
print("Maximum difference:", difference.max().item())
print("Mean difference:   ", difference.mean().item())
