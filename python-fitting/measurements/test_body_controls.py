# body_sliders.py -- MHR semantic body sliders via Jacobian + region-penalised least squares
# Run from the MHR folder:  pixi run python body_sliders.py
# Needs: scipy, trimesh, matplotlib (pixi add scipy trimesh matplotlib)
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull
from mhr.mhr import MHR

# ---------------- config ----------------
NUM_BODY = 20             # identity comps 0..19 are body (head displacement ~0, verified earlier)
STEP = 1.0                # perturbation size for Jacobian / displacement basis
BAND_CM = 1.5             # slice half-thickness (cm)
BETA_LIMIT = 3.0          # clip range for coefficients

# (height fraction from feet, half-width cap as fraction of height)
LOOPS = {
    "chest": (0.72, 0.12),
    "waist": (0.62, 0.15),
    "hips":  (0.52, 0.16),
}
MEAS = ["chest_girth", "waist_girth", "hip_girth", "chest_depth"]
TARGET_CM = [3.0, -3.0, 4.0, 2.0]     # test target for each slider (cm)

# Solver weights
W = 10.0                  # weight on hitting the measurement target
LAM = 1.0                 # weight on keeping protected regions still
LAM_SWEEP = [0.0, 0.1, 1.0, 10.0, 100.0]   # trade-off curve printed for each slider

# Protected regions per slider (mesh coords in cm: x = left/right, y = up, z = depth).
# "Protected" = should NOT move when this slider moves. Tune to taste.
def protect_mask(name, mean):
    x, y = mean[:, 0], mean[:, 1]
    arms_or_head = (np.abs(x) > 28) | (y > 140)
    if name in ("chest_girth", "chest_depth"):
        return arms_or_head | (y < 80)          # leave legs alone
    if name == "waist_girth":
        return arms_or_head | (y < 75)
    if name == "hip_girth":
        return arms_or_head | (y > 115)         # thighs are free to change with hips
    raise ValueError(name)

# ---------------- model ----------------
model = MHR.from_files(device=torch.device("cpu"), lod=1)
faces = np.asarray(model.character.mesh.faces)

def run(betas: np.ndarray) -> np.ndarray:
    """betas [N,45] -> vertices [N,V,3] in cm. Rest pose, no correctives, no expression."""
    b = torch.tensor(betas, dtype=torch.float32)
    pose = torch.zeros(b.shape[0], 204)
    with torch.no_grad():
        v, _ = model(b, pose, None, apply_correctives=False)
    return v.numpy()

mean = run(np.zeros((1, 45)))[0]
y_min, y_max = mean[:, 1].min(), mean[:, 1].max()
H = y_max - y_min
print(f"Mean mesh: {mean.shape[0]} verts, height {H:.1f} cm (Y up)")

# ---------------- fixed vertex loops on the mean mesh ----------------
loop_idx = {}
fig, axes = plt.subplots(1, len(LOOPS), figsize=(5 * len(LOOPS), 5))
for ax, (name, (hf, wf)) in zip(axes, LOOPS.items()):
    y0 = y_min + hf * H
    band = np.abs(mean[:, 1] - y0) < BAND_CM
    sel = band & (np.abs(mean[:, 0]) < wf * H)
    loop_idx[name] = np.where(sel)[0]
    ax.scatter(mean[band, 0], mean[band, 2], s=6, c="lightgray")
    ax.scatter(mean[sel, 0], mean[sel, 2], s=6, c="crimson")
    ax.set_title(f"{name}  y={y0:.0f}cm  n={sel.sum()}")
    ax.set_aspect("equal")
plt.tight_layout(); plt.savefig("slice_check.png", dpi=120); plt.close()

def hull_perimeter(pts2d):
    return ConvexHull(pts2d).area          # 2D: .area is the perimeter

def measure(verts):
    c = verts[loop_idx["chest"]][:, [0, 2]]
    w = verts[loop_idx["waist"]][:, [0, 2]]
    h = verts[loop_idx["hips"]][:, [0, 2]]
    return np.array([
        hull_perimeter(c),
        hull_perimeter(w),
        hull_perimeter(h),
        c[:, 1].max() - c[:, 1].min(),
    ])

m0 = measure(mean)
print("Mean measurements (cm):", dict(zip(MEAS, np.round(m0, 1))))

# ---------------- Jacobian J and displacement basis Bdisp ----------------
# NOTE: named `pert` (not B) so it can't collide with the displacement basis below.
pert = np.zeros((2 * NUM_BODY, 45))
for k in range(NUM_BODY):
    pert[2 * k, k] = +STEP
    pert[2 * k + 1, k] = -STEP
Vp = run(pert)                                            # [2*NUM_BODY, V, 3]

J = np.zeros((len(MEAS), NUM_BODY))
for k in range(NUM_BODY):
    J[:, k] = (measure(Vp[2 * k]) - measure(Vp[2 * k + 1])) / (2 * STEP)

# Vertex displacement (cm) per unit of each component: [V, 3, NUM_BODY]
Bdisp = np.stack([(Vp[2 * k] - Vp[2 * k + 1]) / (2 * STEP) for k in range(NUM_BODY)], axis=-1)

np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("\nJacobian (rows %s; cm per +1 unit):" % MEAS)
print(J)
print("Singular values:", np.linalg.svd(J, compute_uv=False))

# ---------------- solvers ----------------
Jp = np.linalg.pinv(J)                                     # baseline: plain min-norm

def solve_penalised(dm, mask, lam=LAM, w=W):
    Bp = Bdisp[mask].reshape(-1, NUM_BODY)
    Bp = Bp / np.sqrt(len(Bp))                             # size-independent lam
    A = np.vstack([w * J, np.sqrt(lam) * Bp])
    b = np.concatenate([w * dm, np.zeros(len(Bp))])
    return np.linalg.lstsq(A, b, rcond=None)[0]

def to_full(beta):
    full = np.zeros((1, 45))
    full[0, :NUM_BODY] = np.clip(beta, -BETA_LIMIT, BETA_LIMIT)
    return full

normals_mean = trimesh.Trimesh(mean, faces, process=False).vertex_normals

def signed_disp(v1):
    return ((v1 - mean) * normals_mean).sum(1)             # + outward, - inward

# ---------------- per-slider evaluation ----------------
S = np.zeros((NUM_BODY, len(MEAS)))                        # beta per +1 cm of each slider
eye = np.eye(len(MEAS))

for i, name in enumerate(MEAS):
    tgt = TARGET_CM[i]
    dm = eye[i] * tgt
    mask = protect_mask(name, mean)
    print(f"\n=== {name}: target {tgt:+.1f} cm | protected verts: {mask.sum()} ===")

    # trade-off sweep
    print(" lam    | achieved (cm) chest/waist/hip/depth       | max|beta| | protected mean|move| (cm)")
    for lam in LAM_SWEEP:
        beta = solve_penalised(dm, mask, lam=lam)
        v = run(to_full(beta))[0]
        got = measure(v) - m0
        prot = np.linalg.norm(v - mean, axis=1)[mask].mean()
        print(f" {lam:6.1f} | {np.round(got, 2)} | {np.abs(beta).max():9.2f} | {prot:.3f}")

    # baseline vs chosen LAM
    beta_base = Jp @ dm
    beta_pen = solve_penalised(dm, mask, lam=LAM)
    S[:, i] = solve_penalised(eye[i], mask, lam=LAM)       # per-cm slider vector
    v_base = run(to_full(beta_base))[0]
    v_pen = run(to_full(beta_pen))[0]
    s_base, s_pen = signed_disp(v_base), signed_disp(v_pen)
    lim = max(np.abs(s_base).max(), np.abs(s_pen).max())

    fig, axs = plt.subplots(2, 2, figsize=(9, 14))
    for r, (label, s) in enumerate([("plain pinv", s_base), (f"penalised lam={LAM}", s_pen)]):
        for c, (ax_i, view) in enumerate([(0, "front (x-y)"), (2, "side (z-y)")]):
            sc = axs[r, c].scatter(mean[:, ax_i], mean[:, 1], c=s, s=1,
                                   cmap="coolwarm", vmin=-lim, vmax=lim)
            axs[r, c].set_aspect("equal")
            axs[r, c].set_title(f"{name} {tgt:+.0f}cm | {label} | {view}", fontsize=9)
    fig.colorbar(sc, ax=axs, label="signed normal displacement (cm)", shrink=0.6)
    fig.savefig(f"collateral_{name}.png", dpi=120)
    plt.close(fig)
    print(f" saved collateral_{name}.png")

# ---------------- export for the viewer ----------------
# mesh = mean + sum_i cm_i * slider_shapes[..., i]   (cm_i = slider value in cm)
slider_shapes = Bdisp @ S                                  # [V, 3, 4] displacement per +1 cm
np.savez("body_sliders.npz",
         mean=mean.astype(np.float32),
         faces=faces.astype(np.int32),
         slider_shapes=slider_shapes.astype(np.float32),
         slider_names=np.array(MEAS),
         S=S, J=J)
print("\nSaved body_sliders.npz  (mean, faces, slider_shapes[V,3,4], S[20,4], J)")

# sanity check: apply a combined slider set through the real model and compare with linear export
cm = np.array([3.0, -2.0, 2.0, 0.0])
v_model = run(to_full(S @ cm))[0]
v_lin = mean + slider_shapes @ cm
print(f"linear export vs model, max diff: {np.abs(v_model - v_lin).max():.4f} cm  (should be ~0)")
print("Achieved for chest+3 / waist-2 / hip+2:", np.round(measure(v_model) - m0, 2))
