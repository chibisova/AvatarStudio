import torch
import time

from mhr.mhr import MHR


# ==================================================
# 1. Load MHR
# ==================================================

mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

model_parameters = torch.zeros(1, 204)
face_expr_coeffs = torch.zeros(1, 72)


# ==================================================
# 2. Helper function
# ==================================================

def generate_vertices(identity_index, value):

    identity_coeffs = torch.zeros(1, 45)
    identity_coeffs[0, identity_index] = value

    with torch.no_grad():
        vertices, _ = mhr_model(
            identity_coeffs,
            model_parameters,
            face_expr_coeffs
        )

    return vertices[0].clone()


# ==================================================
# 3. Generate neutral mesh
# ==================================================

identity_coeffs = torch.zeros(1, 45)

with torch.no_grad():
    vertices, _ = mhr_model(
        identity_coeffs,
        model_parameters,
        face_expr_coeffs
    )

v_zero = vertices[0].clone()


# ==================================================
# 4. Generate all 45 identity bases
# ==================================================

print("Generating 45 identity bases...")

start = time.perf_counter()

basis = []

for identity_index in range(45):

    v_one = generate_vertices(identity_index, 1.0)

    basis_i = v_one - v_zero

    basis.append(basis_i)

basis = torch.stack(basis)

end = time.perf_counter()

print(f"Basis shape: {basis.shape}")
print(f"Basis generation time: {end - start:.3f} s")


# ==================================================
# 5. Generate random 45-dimensional identity
# ==================================================

torch.manual_seed(42)

random_coeffs = 0.8 * torch.randn(45)

print()
print("Random coefficients:")
print(random_coeffs)


# ==================================================
# 6. Generate actual MHR result
# ==================================================

identity_coeffs = random_coeffs.unsqueeze(0)

with torch.no_grad():

    actual_vertices, _ = mhr_model(
        identity_coeffs,
        model_parameters,
        face_expr_coeffs
    )

actual_vertices = actual_vertices[0]


# ==================================================
# 7. Reconstruct using identity bases
# ==================================================

reconstructed_vertices = v_zero + torch.sum(
    random_coeffs[:, None, None] * basis,
    dim=0
)


# ==================================================
# 8. Compare
# ==================================================

difference = torch.abs(
    actual_vertices - reconstructed_vertices
)

print()
print("==========================================")
print("45-parameter reconstruction test")
print("==========================================")

print("Maximum difference:", difference.max().item())
print("Mean difference:   ", difference.mean().item())
