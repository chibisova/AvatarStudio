import torch
import trimesh
import time
from datetime import datetime

from mhr.mhr import MHR

# Load MHR
mhr_model = MHR.from_files(
    device=torch.device("cpu"),
    lod=1
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# Set random seed based on timestamp
torch.manual_seed(int(timestamp[-6:]))

# Use a different seed each run
#torch.manual_seed(int(time.time()))

# Create random parameters
#torch.manual_seed(0)

# Set identity coefficients to a random value, 
# model parameters to zero (neutral pose), 
# and face expression coefficients to zero (no expression)

# identity_coeffs = 0.8 * torch.randn(1, 45)
# model_parameters = torch.zeros(1, 204)  # Neutral pose
# face_expr_coeffs = torch.zeros(1, 72)  # No expression

# Set identity coefficients to a controlled value

values = [0.0]
#values = [-3.0, 0.0, +3.0]

identity_index = 13

for value in values:

    identity_coeffs = torch.zeros(1, 45)
    identity_coeffs[0, identity_index] = value

    model_parameters = torch.zeros(1, 204)
    face_expr_coeffs = torch.zeros(1, 72)

    # Generate mesh
    with torch.no_grad():
        vertices, skeleton_state = mhr_model(
            identity_coeffs,
            model_parameters,
            face_expr_coeffs
        )
    
    # Print skeleton state information
    print("Skeleton state shape:", skeleton_state.shape)
    print("Skeleton state dtype:", skeleton_state.dtype)
    print("Skeleton:", mhr_model.character.skeleton)

    skeleton = mhr_model.character.skeleton

    print("Skeleton type:", type(skeleton))
    print("Skeleton attributes:")

    for name in dir(skeleton):
        if not name.startswith("_"):
            print(name)
    
    print("Number of joint names:", len(skeleton.joint_names))

    # Print joint names
    for i, name in enumerate(skeleton.joint_names):
        print(i, name)
    
    # Print joint parents
    for i, name in enumerate(skeleton.joint_names):
        print(
            i,
            name,
            "parent =", skeleton.joint_parents[i]
        )

    # Print important joint positions
    important_joints = [
        "root",
        "l_upleg",
        "l_lowleg",
        "l_foot",
        "l_ball",
        "r_upleg",
        "r_lowleg",
        "r_foot",
        "c_spine0",
        "c_spine1",
        "c_spine2",
        "c_spine3",
        "l_clavicle",
        "r_clavicle",
        "l_uparm",
        "r_uparm",
        "l_lowarm",
        "r_lowarm",
        "l_wrist",
        "r_wrist",
        "c_neck",
        "c_head",
    ]

    for name in important_joints:
        index = skeleton.joint_index(name)
        position = skeleton_state[0, index, :3]

        print(f"{name:15s} {index:3d} {position.numpy()}")
    # Save mesh to OBJ file

    vertices_np = vertices[0].numpy()
    faces = mhr_model.character.mesh.faces

    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces,
        process=False
    )

    filename = f"mhr_identity{identity_index}_{value:+.1f}.obj"
    mesh.export(filename)

    print(f"Saved: {filename}")


# Inspect MHR skeleton
print("Skeleton shape:", skeleton_state.shape)

# Print the first few joints
for i in range(min(20, skeleton_state.shape[1])):
    print(i, skeleton_state[0, i])

MHR70_NAMES = {
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_hip",
    10: "right_hip",
    11: "left_knee",
    12: "right_knee",
    13: "left_ankle",
    14: "right_ankle",
    62: "left_wrist",
    41: "right_wrist",
    67: "left_acromion",
    68: "right_acromion",
    69: "neck",
}

for index, name in MHR70_NAMES.items():
    position = skeleton_state[0, index, :3]

    print(
        f"{index:2d}  "
        f"{name:20s} "
        f"{position.numpy()}"
    )

# Calculate distances between joints
import numpy as np

def distance(a, b):
    return np.linalg.norm(a - b)

def pos(name):
    index = skeleton.joint_index(name)
    return skeleton_state[0, index, :3].numpy()

print("Upper arm:", distance(pos("l_uparm"), pos("l_lowarm")))
print("Forearm:", distance(pos("l_lowarm"), pos("l_wrist")))

print("Thigh:", distance(pos("l_upleg"), pos("l_lowleg")))
print("Lower leg:", distance(pos("l_lowleg"), pos("l_foot")))

print("Torso:",
      distance(pos("root"), pos("c_spine0")) +
      distance(pos("c_spine0"), pos("c_spine1")) +
      distance(pos("c_spine1"), pos("c_spine2")) +
      distance(pos("c_spine2"), pos("c_spine3")))

print("Neck:", distance(pos("c_neck"), pos("c_head")))
