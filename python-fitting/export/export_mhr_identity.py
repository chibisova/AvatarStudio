import torch
import struct

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
# 2. Extract MHR data
# ==================================================

base_vertices = torch.tensor(
    blend_shape.base_shape,
    dtype=torch.float32
)

identity_basis = torch.tensor(
    blend_shape.shape_vectors[:45],
    dtype=torch.float32
)

faces = mhr_model.character.mesh.faces


print("Base vertices:", base_vertices.shape)
print("Identity basis:", identity_basis.shape)
print("Faces:", faces.shape)


# ==================================================
# 3. Convert to NumPy
# ==================================================

base_vertices = base_vertices.numpy()
identity_basis = identity_basis.numpy()
faces = faces.astype("int32")


# ==================================================
# 4. Write binary file
# ==================================================

output_path = "mhr_identity.bin"

with open(output_path, "wb") as f:

    # Header
    f.write(struct.pack(
        "<III",
        base_vertices.shape[0],   # vertex count
        identity_basis.shape[0],  # parameter count
        faces.shape[0]             # face count
    ))

    # Base vertices
    f.write(base_vertices.tobytes())

    # Identity basis
    f.write(identity_basis.tobytes())

    # Triangle indices
    f.write(faces.tobytes())


print()
print("Saved:", output_path)
