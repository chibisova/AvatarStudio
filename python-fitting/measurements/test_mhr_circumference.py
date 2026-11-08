"""
MHR Mesh Circumference Measurement — Geometry Prototype
========================================================

PURPOSE
    Experiments with measuring body circumferences directly
    from an MHR triangle mesh.

WHAT IT DOES
    1. Generates a neutral MHR body mesh.
    2. Defines a measurement plane using skeleton joints.
    3. Intersects the plane with the triangle mesh.
    4. Builds an intersection graph and extracts closed loops.
    5. Selects the loop containing a specified 3D point.
    6. Calculates the loop perimeter as circumference.

CURRENT STATUS
    Working prototype. Waist, bust, and hip slices have been
    tested using approximate skeleton-based plane definitions.

IMPORTANT
    This is an experimental geometry script, not yet a
    production measurement system. Anatomical plane placement
    and measurement accuracy still need validation.

ROLE IN PROJECT
    Reference implementation for the shared measurement engine.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mhr.mhr import MHR
from measurement_engine import measure_circumference
from measurement_engine import (
    mesh_plane_intersections,
    build_intersection_graph,
    extract_loops,
    loop_perimeter,
    find_loop_containing_point,
    measure_circumference,
)


# ============================================================
# 1. Load MHR and generate neutral mesh
# ============================================================

mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

faces = mhr_model.character.mesh.faces

identity_coeffs = torch.zeros(1, 45)
model_parameters = torch.zeros(1, 204)
face_expr_coeffs = torch.zeros(1, 72)

with torch.no_grad():
    vertices, skeleton_state = mhr_model(
        identity_coeffs,
        model_parameters,
        face_expr_coeffs
    )

vertices = vertices[0].numpy()


# ============================================================
# 2. Skeleton helpers
# ============================================================

skeleton = mhr_model.character.skeleton

def get_joint_position(name):
    index = skeleton.joint_index(name)
    return skeleton_state[0, index, :3].numpy()


print("Number of joints:", skeleton.size)

for name in [
    "c_spine0",
    "c_spine1",
    "c_spine2",
    "c_spine3",
]:
    print(
        f"{name}: index={skeleton.joint_index(name)}, "
        f"position={get_joint_position(name)}"
    )


# ============================================================
# 3. Define waist measurement plane
# ============================================================

# For now:
#   position = midpoint between c_spine1 and c_spine2
#   normal   = direction from c_spine1 to c_spine2

waist_lower = get_joint_position("c_spine1")
waist_upper = get_joint_position("c_spine2")

plane_origin = (waist_lower + waist_upper) / 2.0

plane_normal = waist_upper - waist_lower
plane_normal /= np.linalg.norm(plane_normal)

#plane_origin = (waist_lower + waist_upper) / 2.0
#plane_normal = np.array([0.0, 1.0, 0.0])
#selected_point = plane_origin.copy()

print("\n--- Waist plane ---")
print("Plane origin:", plane_origin)
print("Plane normal:", plane_normal)


# ============================================================
# 4. Mesh / plane intersection
# ============================================================

def edge_plane_intersection(
    a,
    b,
    plane_origin,
    plane_normal
):
    edge = b - a

    denominator = np.dot(plane_normal, edge)

    if abs(denominator) < 1e-8:
        return None

    t = np.dot(
        plane_normal,
        plane_origin - a
    ) / denominator

    if t < 0.0 or t > 1.0:
        return None

    return a + t * edge


def triangle_plane_intersections(
    v0,
    v1,
    v2,
    plane_origin,
    plane_normal
):
    edges = [
        (v0, v1),
        (v1, v2),
        (v2, v0),
    ]

    points = []

    for a, b in edges:
        point = edge_plane_intersection(
            a,
            b,
            plane_origin,
            plane_normal
        )

        if point is not None:
            points.append(point)

    # Remove duplicate intersections within one triangle
    unique = []

    for point in points:
        if not any(
            np.linalg.norm(point - p) < 1e-6
            for p in unique
        ):
            unique.append(point)

    return unique



segments = mesh_plane_intersections(
    vertices,
    faces,
    plane_origin,
    plane_normal
)

print("\nNumber of intersection segments:", len(segments))


# ============================================================
# 5. Build intersection graph
# ============================================================

unique_points, segment_ids, adjacency = (
    build_intersection_graph(
        segments,
        tolerance=1e-3
    )
)

print("Total intersection points:", len(segments) * 2)
print("Unique points:", len(unique_points))


# ============================================================
# 6. Validate graph topology
# ============================================================

degree = Counter()

for p0_id, p1_id in segment_ids:

    degree[p0_id] += 1
    degree[p1_id] += 1


print("\n--- Graph topology ---")

for d, count in sorted(
    Counter(degree.values()).items()
):
    print(
        f"degree {d}: {count} points"
    )


invalid_points = [
    (point_id, d)
    for point_id, d in degree.items()
    if d != 2
]

if invalid_points:

    print(
        "\nWARNING: intersection graph "
        "contains non-degree-2 points."
    )

    for point_id, d in invalid_points[:20]:

        print(
            f"  point {point_id}: "
            f"degree={d}, "
            f"position={unique_points[point_id]}"
        )

else:

    print(
        "All intersection points have degree 2."
    )


# ============================================================
# 7. Extract closed loops
# ============================================================

loops = extract_loops(adjacency)

print("\nNumber of closed loops:", len(loops))


# ============================================================
# 8. Measure loops
# ============================================================


for i, loop in enumerate(loops):

    perimeter = loop_perimeter(
        loop,
        unique_points
    )

    print(
        f"Loop {i}: "
        f"{len(loop)} points, "
        f"{perimeter:.2f} cm"
    )

# ============================================================
# 9. Select loop containing a point
# ============================================================

def project_point_to_plane(
    point,
    plane_origin,
    plane_normal
):
    distance = np.dot(
        point - plane_origin,
        plane_normal
    )

    return point - distance * plane_normal


def point_inside_loop(
    point_2d,
    loop_2d
):
    """
    Ray-casting point-in-polygon test.
    """

    x, y = point_2d

    inside = False

    j = len(loop_2d) - 1

    for i in range(len(loop_2d)):

        xi, yi = loop_2d[i]
        xj, yj = loop_2d[j]

        intersects = (
            (yi > y) != (yj > y)
            and
            x < (
                (xj - xi)
                * (y - yi)
                / (yj - yi)
                + xi
            )
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def build_plane_basis(plane_normal):
    """
    Construct two orthonormal axes lying inside the plane.
    """

    n = plane_normal / np.linalg.norm(plane_normal)

    # Choose a reference vector that isn't parallel to n
    if abs(n[0]) < 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    else:
        reference = np.array([0.0, 1.0, 0.0])

    # First axis in the plane
    u = np.cross(n, reference)
    u /= np.linalg.norm(u)

    # Second axis in the plane
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    return u, v


def project_points_to_plane_2d(
    points,
    plane_origin,
    plane_normal
):
    """
    Convert 3D points into 2D coordinates
    relative to the measurement plane.
    """

    u, v = build_plane_basis(plane_normal)

    relative = points - plane_origin

    x = relative @ u
    y = relative @ v

    return np.column_stack((x, y))

# ============================================================
# 10. Test waist, bust, and hip measurements
# ============================================================

MEASUREMENTS = {
    "waist": ("c_spine1", "c_spine2"),
    "bust":  ("c_spine2", "c_spine3"),
    "hip":   ("c_spine0", "c_spine1"),
}

measurement_results = {}

measurement_visuals = {}

for name, (lower_name, upper_name) in MEASUREMENTS.items():

    lower = get_joint_position(lower_name)
    upper = get_joint_position(upper_name)

    plane_origin = (lower + upper) / 2.0
    """
    cut waist plane normal to be horizontal (Y up) for more accurate circumference measurement
    
    if name == "waist":
        plane_normal = np.array([0.0, 1.0, 0.0])
    else:
        plane_normal = upper - lower
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
    """
    plane_normal = upper - lower
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    selected_point = plane_origin.copy()

    circumference, info = measure_circumference(
        vertices=vertices,
        faces=faces,
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        selected_point=selected_point,
    )

    print(f"\n{name} plane normal:", plane_normal)
    print(f"{name} plane origin:", plane_origin)

    measurement_results[name] = circumference

    if circumference is None:
        print(f"{name}: FAILED — {info}")
    else:
        measurement_visuals[name] = {
            "circumference": circumference,
            "contour_points": info["contour_points"],
            "plane_origin": plane_origin.copy(),
            "plane_normal": (
                plane_normal / np.linalg.norm(plane_normal)
            ),
        }

        print(
            f"{name}: {circumference:.2f} cm "
            f"(loop {info['loop_index']}, "
            f"{info['loop_count']} loops)"
        )

    print("\n--- Measurement summary ---")
    for name, value in measurement_results.items():
        if value is not None:
            print(f"{name}: {value:.2f} cm")
        else:
            print(f"{name}: unavailable")

contour = measurement_visuals["waist"]["contour_points"]

print("Waist contour Y min:", contour[:, 1].min())
print("Waist contour Y max:", contour[:, 1].max())
print("Waist contour Y range:", np.ptp(contour[:, 1]))

# ============================================================
# 11. Interactive 3D visualization
# ============================================================

from matplotlib.widgets import Button

COLORS = {
    "waist": "red",
    "bust": "green",
    "hip": "darkorange",
}

fig = plt.figure(figsize=(11, 10))

# Reserve a dedicated bottom row for controls
fig.subplots_adjust(
    left=0.08,
    right=0.92,
    top=0.90,
    bottom=0.22,
)

ax = fig.add_subplot(111, projection="3d")

# MHR uses Y as vertical.
# Plot coordinates as (X, Z, Y) so Matplotlib's Z axis is vertical.
def to_plot(points):
    points = np.asarray(points)
    return points[..., [0, 2, 1]]

# Avatar mesh
plot_vertices = to_plot(vertices)
triangles = plot_vertices[faces]

mesh_collection = Poly3DCollection(
    triangles,
    facecolor="lightsteelblue",
    edgecolor="none",
    alpha=0.22,
)
ax.add_collection3d(mesh_collection)

# Measurement contours
for name, data in measurement_visuals.items():
    contour = to_plot(data["contour_points"])
    contour = np.vstack([contour, contour[0]])

    ax.plot(
        contour[:, 0],
        contour[:, 1],
        contour[:, 2],
        color=COLORS.get(name, "black"),
        linewidth=3,
        label=f"{name.title()}: {data['circumference']:.2f} cm",
    )

# Equal spatial scale
mins = plot_vertices.min(axis=0)
maxs = plot_vertices.max(axis=0)
center = (mins + maxs) / 2
radius = (maxs - mins).max() / 2

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("X")
ax.set_ylabel("Z (depth)")
ax.set_zlabel("Y (height)")
ax.set_title("MHR Body Measurements")
ax.legend(loc="upper left")

# Camera state
camera = {
    "yaw": 90.0,
    "pitch": 0.0,
    "roll": 0.0,
}

def apply_camera():
    ax.view_init(
        elev=camera["pitch"],
        azim=camera["yaw"],
        roll=camera["roll"],
    )
    fig.canvas.draw_idle()

def change_camera(axis, delta):
    camera[axis] += delta

    if axis == "pitch":
        camera[axis] = np.clip(camera[axis], -89, 89)

    apply_camera()

def reset_camera(event):
    camera.update(yaw=90.0, pitch=0.0, roll=0.0)
    apply_camera()

# Each button gets its own space below the plot.
button_specs = [
    ("Yaw -", lambda e: change_camera("yaw", -10)),
    ("Yaw +", lambda e: change_camera("yaw", 10)),
    ("Pitch -", lambda e: change_camera("pitch", -10)),
    ("Pitch +", lambda e: change_camera("pitch", 10)),
    ("Roll -", lambda e: change_camera("roll", -10)),
    ("Roll +", lambda e: change_camera("roll", 10)),
    ("Reset", reset_camera),
]

buttons = []

left_start = 0.08
button_width = 0.105
gap = 0.018
button_bottom = 0.09
button_height = 0.055

for i, (label, callback) in enumerate(button_specs):
    left = left_start + i * (button_width + gap)
    button_ax = fig.add_axes([
        left, button_bottom, button_width, button_height
    ])
    button = Button(button_ax, label)
    button.on_clicked(callback)
    buttons.append(button)

apply_camera()
plt.show()
