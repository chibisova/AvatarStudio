
"""
MHR Inverse Fitting — Prototype

PURPOSE
    Test how MHR identity coefficients affect body circumferences.

NEXT STEPS
    1. Generate a mesh from identity coefficients.
    2. Measure waist, bust, and hip.
    3. Estimate a measurement Jacobian.
    4. Optimize identity coefficients toward target measurements.
"""

import numpy as np
import torch
import time

from mhr.mhr import MHR
from measurement_engine import measure_circumference
import matplotlib.pyplot as plt
import contextlib
import io


# ------------------------------------------------------------
# 1. Initialize MHR
# ------------------------------------------------------------

device = torch.device("cpu")

mhr_model = MHR.from_files(
    device=device,
    lod=1,
)

faces = mhr_model.character.mesh.faces
skeleton = mhr_model.character.skeleton

model_parameters = torch.zeros(1, 204, device=device)
face_expr_coeffs = torch.zeros(1, 72, device=device)


# ------------------------------------------------------------
# 2. Generate mesh from identity coefficients
# ------------------------------------------------------------

def generate_mesh(identity_coeffs):
    """Generate mesh vertices and skeleton state for given identity coefficients."""

    identity_tensor = torch.as_tensor(
        identity_coeffs,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 45)

    with torch.no_grad():
        vertices, skeleton_state = mhr_model(
            identity_tensor,
            model_parameters,
            face_expr_coeffs,
        )

    vertices = vertices[0].detach().cpu().numpy()
    skeleton_state = skeleton_state[0].detach().cpu().numpy()

    return vertices, skeleton_state


# ------------------------------------------------------------
# 3. Measure waist, bust, and hip
# ------------------------------------------------------------

MEASUREMENTS = {
    "waist": ("c_spine1", "c_spine2"),
    "bust": ("c_spine2", "c_spine3"),
    "hip": ("c_spine0", "c_spine1"),
}


def measure_body(identity_coeffs):
    """Return body circumferences for a given identity vector."""

    total_start = time.perf_counter()
    mesh_start = time.perf_counter()

    vertices, skeleton_state = generate_mesh(identity_coeffs)
    print(
        f"  Mesh generation: "
        f"{time.perf_counter() - mesh_start:.3f} s"
    )
    results = {}

    for name, (lower_name, upper_name) in MEASUREMENTS.items():
        lower_idx = skeleton.joint_index(lower_name)
        upper_idx = skeleton.joint_index(upper_name)

        lower = skeleton_state[lower_idx, :3]
        upper = skeleton_state[upper_idx, :3]

        plane_origin = (lower + upper) / 2.0
        plane_normal = upper - lower

        norm = np.linalg.norm(plane_normal)
        if norm < 1e-8:
            results[name] = None
            continue

        plane_normal = plane_normal / norm

        measurement_start = time.perf_counter()
        circumference, info = measure_circumference(
            vertices=vertices,
            faces=faces,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            selected_point=plane_origin,
        )
        print(
            f"  {name} measurement: "
            f"{time.perf_counter() - measurement_start:.3f} s"
        )
        results[name] = circumference

        if circumference is None:
            print(f"Warning: {name} measurement failed: {info}")
        
    print(
            f"  Total measure_body: "
            f"{time.perf_counter() - total_start:.3f} s"
        )
    return results



# ------------------------------------------------------------
# 4. Estimate measurement Jacobian
# ------------------------------------------------------------

MEASUREMENT_NAMES = ["waist", "bust", "hip"]
NUM_BODY_COEFFS = 20


def estimate_jacobian(identity_coeffs, delta=0.01):
    """
    Estimate measurement sensitivity using forward finite differences.

    Returns:
        jacobian: array with shape (3, 20) for [waist, bust, hip] vs. 20 body coefficients.
        baseline: baseline measurement dictionary
    """

    print("Estimating Jacobian around identity:", identity_coeffs[:5])
    baseline = measure_body(identity_coeffs)

    if any(baseline[name] is None for name in MEASUREMENT_NAMES):
        raise RuntimeError("Baseline measurement failed.")

    jacobian = np.zeros(
        (len(MEASUREMENT_NAMES), NUM_BODY_COEFFS)
    )

    for j in range(NUM_BODY_COEFFS):
        perturbed_identity = identity_coeffs.copy()
        perturbed_identity[j] += delta

        perturbed = measure_body(perturbed_identity)

        for i, name in enumerate(MEASUREMENT_NAMES):
            if perturbed[name] is None:
                raise RuntimeError(
                    f"Measurement '{name}' failed for coefficient {j}."
                )

            jacobian[i, j] = (
                perturbed[name] - baseline[name]
            ) / delta

    return jacobian, baseline

# ------------------------------------------------------------
# 5. Compute a regularized identity update
# ------------------------------------------------------------

def solve_identity_update(jacobian, current, target, regularization=1.0):
    """
    Compute a regularized least-squares update for identity coefficients.

    Args:
        jacobian: Measurement Jacobian with shape (3, 20).
        current: Current measurements in [waist, bust, hip] order.
        target: Target measurements in the same order.
        regularization: Penalty on the magnitude of the update.

    Returns:
        Parameter update with shape (20,).
    """

    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)

    residual = target - current

    lhs = jacobian.T @ jacobian
    lhs += regularization * np.eye(jacobian.shape[1])

    rhs = jacobian.T @ residual

    return np.linalg.solve(lhs, rhs)


# ------------------------------------------------------------
# 6. Iterative identity fitting
# ------------------------------------------------------------

def fit_identity(
    initial_identity,
    target,
    max_iterations=10,
    tolerance=0.01,
    regularization=1.0,
    max_step_norm=0.1,
):
    """
    Iteratively fit MHR identity coefficients to target circumferences.

    Target order: waist, bust, hip.
    """

    identity = initial_identity.astype(np.float64).copy()
    target = np.asarray(target, dtype=np.float64)

    for iteration in range(max_iterations):
        iteration_start = time.perf_counter()
        measurements = measure_body(identity)

        if any(measurements[name] is None for name in MEASUREMENT_NAMES):
            raise RuntimeError("Current body measurement failed.")

        current = np.array([
            measurements[name] for name in MEASUREMENT_NAMES
        ])

        residual = target - current
        error = np.linalg.norm(residual)

        print(
            f"Iteration {iteration}: "
            f"error={error:.4f} cm, "
            f"waist={current[0]:.3f}, "
            f"bust={current[1]:.3f}, "
            f"hip={current[2]:.3f}"
        )

        if error < tolerance:
            break

        jacobian_start = time.perf_counter()

        print(f"Fitting iteration {iteration}")
        jacobian, _ = estimate_jacobian(identity)
        print("Jacobian shape:", jacobian.shape)
        print("Jacobian rank:", np.linalg.matrix_rank(jacobian))
        print("Jacobian column norms:")
        print(np.linalg.norm(jacobian, axis=0))

        print(
            f"  Jacobian time: "
            f"{time.perf_counter() - jacobian_start:.3f} s"
        )

        # Calculate the proposed update.
        delta_identity = solve_identity_update(
            jacobian,
            current,
            target,
            regularization=regularization,
        )

        print(f"  Regularization: {regularization}")


        proposed_norm = np.linalg.norm(delta_identity)
        print(f"  Proposed update norm: {proposed_norm:.8f}")

        # Limit the update size for stability.
        if proposed_norm > max_step_norm:
            delta_identity *= max_step_norm / proposed_norm

        clipped_norm = np.linalg.norm(delta_identity)
        print(f"  Update norm after clipping: {clipped_norm:.8f}")

        # Try the proposed update. If it increases the error,
        # halve the step until it improves or becomes too small.
        accepted = False
        scale = 1.0
        
        line_search_start = time.perf_counter()

        while scale >= 1.0 / 128.0:
            # candidate = identity + scale * delta_identity # works with 45 coefficients, but we only want to update the first 20
            candidate = identity.copy()
            candidate[:NUM_BODY_COEFFS] += scale * delta_identity
            candidate_measurements = measure_body(candidate)

            if any(
                candidate_measurements[name] is None
                for name in MEASUREMENT_NAMES
            ):
                scale *= 0.5
                continue

            candidate_values = np.array([
                candidate_measurements[name]
                for name in MEASUREMENT_NAMES
            ])

            candidate_error = np.linalg.norm(target - candidate_values)

            print(
                f"    scale={scale:.6f}, "
                f"candidate waist={candidate_values[0]:.3f}, "
                f"candidate error={candidate_error:.4f}"
            )

            if candidate_error < error:
                identity = candidate
                accepted = True
                break

            scale *= 0.5

        print(
            f"  Line search time: "
            f"{time.perf_counter() - line_search_start:.3f} s"
        )
        if not accepted:
            print("No improving update found; stopping.")
            break
                
        iteration_time = time.perf_counter() - iteration_start
        print(f"  Iteration computation time: {iteration_time:.3f} s")


    final_measurements = measure_body(identity)

    return identity, final_measurements


def export_mesh_obj(filepath, vertices, faces):
    """Export vertices and triangular faces to an OBJ file."""
    with open(filepath, "w") as f:
        for vertex in vertices:
            f.write(
                f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"
            )

        # OBJ indices are 1-based; NumPy face indices are 0-based.
        for face in faces:
            f.write(
                f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n"
            )

    print(f"Exported: {filepath}")


# ------------------------------------------------------------
# Main execution for testing
# ------------------------------------------------------------

if __name__ == "__main__":
    initial_identity = np.zeros(45, dtype=np.float32)

    # Measure the neutral body.
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = measure_body(initial_identity)

    baseline_values = np.array([
        baseline[name] for name in MEASUREMENT_NAMES
    ])

    # Target: waist +X cm, bust +X cm, hip +X cm .
    target_offset = np.array([5.0, 10.0, 15.0])
    target = baseline_values + target_offset

    # Test regularization values across several orders of magnitude.
    regularization = 0.01

    # Store results.
    fitted_values = []
    identity_changes = []
    measurement_errors = []

    print("Baseline measurements:")
    for name, value in zip(MEASUREMENT_NAMES, baseline_values):
        print(f"  {name}: {value:.3f} cm")

    print("\nRunning regularization experiment...")

    print(f"\nRegularization = {regularization:g}")

    # Every run starts from the same neutral identity.
    fitted_identity, final_measurements = fit_identity(
        initial_identity=initial_identity,
        target=target,
        max_iterations=50,
        tolerance=0.01,
        regularization=regularization,
        max_step_norm=0.1,
    )

    values = np.array([
        final_measurements[name]
        for name in MEASUREMENT_NAMES
    ])

    measurement_error = np.linalg.norm(target - values)
    identity_change = np.linalg.norm(
        fitted_identity - initial_identity
    )

    fitted_values.append(values)
    measurement_errors.append(measurement_error)
    identity_changes.append(identity_change)

    print("  Final measurements:")
    for name, value, target_value in zip(
        MEASUREMENT_NAMES, values, target
    ):
        print(
            f"    {name}: {value:.3f} cm "
            f"(target {target_value:.3f}, "
            f"error {value - target_value:+.3f} cm)"
        )

    print(f"  Total measurement error: {measurement_error:.6f} cm")
    print(f"  Identity change: {identity_change:.6f}")

    fitted_values = np.array(fitted_values)


    # Generate the neutral mesh.
    neutral_vertices, _ = generate_mesh(initial_identity)

    # Generate the fitted mesh.
    fitted_vertices, _ = generate_mesh(fitted_identity)


    displacement = np.linalg.norm(
        fitted_vertices - neutral_vertices,
        axis=1,
    )

    print("\n=== Mesh comparison ===")
    print("Identity coefficient change:",
        np.linalg.norm(fitted_identity - initial_identity))
    print("Maximum vertex displacement (model units):",
        displacement.max())
    print("Mean vertex displacement (model units):",
        displacement.mean())
    print("Vertices moved:",
        np.count_nonzero(displacement > 1e-5),
        "/", len(displacement))

    print("\nIdentity coefficient changes:")
    print(fitted_identity[:20] - initial_identity[:20])


    export_mesh_obj(
        "mhr_neutral.obj",
        neutral_vertices,
        faces,
    )

    export_mesh_obj(
        "mhr_fitted.obj",
        fitted_vertices,
        faces,
    )
    

    # Save the optimized identity coefficients too.
    np.save("mhr_fitted_identity.npy", fitted_identity)




"""
# ------------------------------------------------------------
# Main execution for testing
# 1. Test the fitting process with a target change in measurements.
# 2. Explore the effect of different regularization values on the fitting results.
# 3. Plot the measurement changes and identity update magnitudes for analysis.
# 4. Print a summary table of regularization vs. measurement error and identity change.
# ------------------------------------------------------------

if __name__ == "__main__":
    initial_identity = np.zeros(45, dtype=np.float32)

    # Measure the neutral body.
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = measure_body(initial_identity)

    baseline_values = np.array([
        baseline[name] for name in MEASUREMENT_NAMES
    ])

    # Target: waist +1.0 cm, bust +0.5 cm, hip unchanged.
    target_offset = np.array([1.0, 0.5, 0.0])
    target = baseline_values + target_offset

    # Test regularization values across several orders of magnitude.
    regularization_values = [
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0
    ]

    # Store results.
    fitted_values = []
    identity_changes = []
    measurement_errors = []

    print("Baseline measurements:")
    for name, value in zip(MEASUREMENT_NAMES, baseline_values):
        print(f"  {name}: {value:.3f} cm")

    print("\nRunning regularization experiment...")

    for reg in regularization_values:
        print(f"\nRegularization = {reg:g}")

        # Every run starts from the same neutral identity.
        with contextlib.redirect_stdout(io.StringIO()):
            fitted_identity, final_measurements = fit_identity(
                initial_identity=initial_identity,
                target=target,
                max_iterations=1,
                tolerance=0.01,
                regularization=reg,
                max_step_norm=1.0,
            )

        values = np.array([
            final_measurements[name]
            for name in MEASUREMENT_NAMES
        ])

        measurement_error = np.linalg.norm(target - values)
        identity_change = np.linalg.norm(
            fitted_identity - initial_identity
        )

        fitted_values.append(values)
        measurement_errors.append(measurement_error)
        identity_changes.append(identity_change)

        print("  Final measurements:")
        for name, value, target_value in zip(
            MEASUREMENT_NAMES, values, target
        ):
            print(
                f"    {name}: {value:.3f} cm "
                f"(target {target_value:.3f}, "
                f"error {value - target_value:+.3f} cm)"
            )

        print(f"  Total measurement error: {measurement_error:.6f} cm")
        print(f"  Identity change: {identity_change:.6f}")

    fitted_values = np.array(fitted_values)

    # Plot measurement changes relative to the neutral body.
    measurement_changes = fitted_values - baseline_values

    plt.figure(figsize=(9, 5))

    for i, name in enumerate(MEASUREMENT_NAMES):
        plt.plot(
            regularization_values,
            measurement_changes[:, i],
            marker="o",
            label=name.capitalize(),
        )

        plt.axhline(
            target_offset[i],
            linestyle="--",
            alpha=0.5,
        )

    plt.xscale("log")
    plt.xlabel("Regularization (lambda)")
    plt.ylabel("Measurement change from baseline (cm)")
    plt.title("Effect of Regularization on Body Measurements")
    plt.xticks(
        regularization_values,
        [f"{value:g}" for value in regularization_values],
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot identity update magnitude.
    plt.figure(figsize=(8, 4))
    plt.plot(
        regularization_values,
        identity_changes,
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Regularization (lambda)")
    plt.ylabel("Identity update norm")
    plt.title("Effect of Regularization on Identity Change")
    plt.xticks(
        regularization_values,
        [f"{value:g}" for value in regularization_values],
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot total measurement error.
    plt.figure(figsize=(8, 4))
    plt.plot(
        regularization_values,
        measurement_errors,
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Regularization (lambda)")
    plt.ylabel("Total measurement error (cm)")
    plt.title("Measurement Error vs. Regularization")
    plt.xticks(
        regularization_values,
        [f"{value:g}" for value in regularization_values],
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print a compact comparison table.
    print("\n--- Regularization comparison ---")
    print(
        f"{'Lambda':>10} | {'Error (cm)':>12} | "
        f"{'Identity change':>16}"
    )
    print("-" * 45)

    for reg, error, change in zip(
        regularization_values,
        measurement_errors,
        identity_changes,
    ):
        print(f"{reg:10.5g} | {error:12.6f} | {change:16.6f}")

"""
