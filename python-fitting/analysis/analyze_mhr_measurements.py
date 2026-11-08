import struct
import numpy as np


PATH = "mhr_identity.bin"


with open(PATH, "rb") as f:
    vertex_count, parameter_count, face_count = struct.unpack(
        "<III",
        f.read(12)
    )

    base_vertices = np.frombuffer(
        f.read(vertex_count * 3 * 4),
        dtype=np.float32
    ).reshape(vertex_count, 3)

    identity_basis = np.frombuffer(
        f.read(parameter_count * vertex_count * 3 * 4),
        dtype=np.float32
    ).reshape(parameter_count, vertex_count, 3)

    faces = np.frombuffer(
        f.read(face_count * 3 * 4),
        dtype=np.int32
    ).reshape(face_count, 3)


print("Vertices:", vertex_count)
print("Parameters:", parameter_count)
print("Faces:", face_count)
print("Base vertices:", base_vertices.shape)
print("Identity basis:", identity_basis.shape)
print("Faces:", faces.shape)

height = (
    base_vertices[:, 1].max()
    - base_vertices[:, 1].min()
)

print("Height:", height)

y_min = base_vertices[:, 1].min()
y_max = base_vertices[:, 1].max()

print("Y min:", y_min)
print("Y max:", y_max)


# Inspect body width at different heights

for height_percent in [0.10, 0.15, 0.20, .25, .30, .35, .40, .45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
    y = y_min + (y_max - y_min) * height_percent

    tolerance = 1.0

    mask = np.abs(base_vertices[:, 1] - y) < tolerance

    if np.any(mask):
        x = base_vertices[mask, 0]

        width = x.max() - x.min()

        print(
            f"{height_percent * 100:.0f}% height: "
            f"Y={y:.2f}, "
            f"vertices={len(x)}, "
            f"width={width:.2f} cm"
        )


# X-Y front projection of the mesh

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 10))

plt.scatter(
    base_vertices[:, 0],
    base_vertices[:, 1],
    s=1
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("MHR body - front projection")

plt.axis("equal")
plt.show()

