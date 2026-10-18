from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def normalize_vectors(
    vectors: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize vectors along the specified axis.
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norms, eps)


def depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Back-project a depth image to 3D camera-frame points.

    Parameters
    ----------
    depth:
        Array with shape [H, W], in meters.
        Assumes depth represents camera-frame z coordinate.
    intrinsics:
        Camera intrinsic parameters.

    Returns
    -------
    points:
        Array with shape [H, W, 3].
    """
    if depth.ndim != 2:
        raise ValueError(
            f"depth must have shape [H, W], got {depth.shape}"
        )

    height, width = depth.shape

    pixel_v, pixel_u = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )

    z = depth.astype(np.float64, copy=False)

    x = (pixel_u - intrinsics.cx) * z / intrinsics.fx
    y = (pixel_v - intrinsics.cy) * z / intrinsics.fy

    return np.stack([x, y, z], axis=-1)


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Apply a 4x4 rigid transformation to 3D points.

    Parameters
    ----------
    points:
        Array with final dimension 3, e.g. [H, W, 3].
    transform:
        Homogeneous transformation matrix [4, 4].

    Returns
    -------
    transformed_points:
        Array with the same shape as points.
    """
    transform = np.asarray(transform, dtype=np.float64)

    if transform.shape != (4, 4):
        raise ValueError(
            f"transform must have shape [4, 4], got {transform.shape}"
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    return np.einsum(
        "ij,...j->...i",
        rotation,
        points,
    ) + translation


def make_tangent_basis(
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct two orthonormal tangent vectors for each unit vector mu.

    Parameters
    ----------
    mu:
        Array with shape [N, 3].

    Returns
    -------
    b1, b2:
        Arrays with shape [N, 3].

        For each candidate:
            b1 dot mu = 0
            b2 dot mu = 0
            b1 dot b2 = 0
    """
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError(
            f"mu must have shape [N, 3], got {mu.shape}"
        )

    reference = np.zeros_like(mu)

    # Use z-axis unless mu is nearly parallel to it.
    use_z_axis = np.abs(mu[:, 2]) < 0.9

    reference[use_z_axis] = np.array([0.0, 0.0, 1.0])
    reference[~use_z_axis] = np.array([0.0, 1.0, 0.0])

    b1 = np.cross(mu, reference)
    b1 = normalize_vectors(b1)

    b2 = np.cross(mu, b1)
    b2 = normalize_vectors(b2)

    return b1, b2


def sphere_log_map(
    mu: np.ndarray,
    g: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute the spherical logarithm map Log_mu(g) on S^2.

    Parameters
    ----------
    mu:
        Base directions with shape [N, 3].
    g:
        Observed direction with shape [3].
    eps:
        Numerical stability constant.

    Returns
    -------
    tangent_vectors:
        Array with shape [N, 3].
    """
    mu = normalize_vectors(mu)

    g = np.asarray(g, dtype=np.float64)

    if g.shape != (3,):
        raise ValueError(
            f"g must have shape [3], got {g.shape}"
        )

    g = normalize_vectors(g)

    cosine = np.sum(mu * g[None, :], axis=1)
    cosine = np.clip(cosine, -1.0 + eps, 1.0 - eps)

    theta = np.arccos(cosine)
    sin_theta = np.sin(theta)

    tangent_component = (
        g[None, :] - cosine[:, None] * mu
    )

    scale = np.ones_like(theta)

    non_small = theta >= 1e-5
    scale[non_small] = (
        theta[non_small]
        / np.maximum(sin_theta[non_small], eps)
    )

    return scale[:, None] * tangent_component


def logsumexp(
    values: np.ndarray,
) -> float:
    """
    Numerically stable log(sum(exp(values))).
    """
    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        return -np.inf

    maximum = np.max(values[finite_mask])

    return maximum + np.log(
        np.sum(np.exp(values[finite_mask] - maximum))
    )


def pixel_posterior_from_gesture(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    eye_position: np.ndarray,
    finger_position: np.ndarray,
    observed_direction: np.ndarray,
    sigma_u: float,
    alpha_depth: float,
    camera_to_world: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    log_prior: Optional[np.ndarray] = None,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
    covariance_jitter: float = 1e-8,
    fitting_joint_coordinates: Optional[np.ndarray] = None,
    fitting_covariance_beta: float = 1.0,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Compute a posterior probability for every valid RGB-D pixel.

    Model
    -----
    For each pixel candidate x_i:

        d_i = ||x_i - E||

        u_i = (x_i - E) / d_i

        mu_i = (x_i - F) / ||x_i - F||

        sigma_d_i = alpha_depth * d_i

        Sigma_y_i =
            sigma_d_i^2 u_i u_i^T
            +
            d_i^2 sigma_u^2 (I - u_i u_i^T)

        Sigma_g_i =
            1 / r_i^2
            P_i Sigma_y_i P_i

    where

        r_i = ||x_i - F||

        P_i = I - mu_i mu_i^T.

    The covariance is then represented in the two-dimensional
    tangent plane of mu_i.

    Parameters
    ----------
    depth:
        Depth image [H, W], in meters.
    intrinsics:
        Camera intrinsics.
    eye_position:
        Eye position E, shape [3].
    finger_position:
        Finger position F, shape [3].
    observed_direction:
        Observed gesture direction g_obs, shape [3].
    sigma_u:
        Angular uncertainty standard deviation, in radians.
    alpha_depth:
        Relative depth uncertainty parameter:

            sigma_d = alpha_depth * d

    camera_to_world:
        Optional [4, 4] matrix mapping camera coordinates to world
        coordinates. If supplied, E, F and g_obs must be in world coordinates.
    valid_mask:
        Optional Boolean array [H, W]. Only True pixels are candidates.
    log_prior:
        Optional array [H, W] containing per-pixel log-priors.
    min_depth:
        Minimum valid depth in meters.
    max_depth:
        Maximum valid depth in meters.
    covariance_jitter:
        Small positive number added to each 2x2 covariance matrix.
    fitting_joint_coordinates:
        Three or four index-finger joint coordinates [N, 3]. When supplied, PCA is
        used to fit the pointing direction and its anisotropic covariance.
    fitting_covariance_beta:
        Non-negative calibration coefficient for the PCA fitting covariance.
    device:
        ``cpu`` uses NumPy, ``cuda`` requires PyTorch CUDA, and ``auto`` uses
        CUDA when available while falling back to NumPy.

    Returns
    -------
    result:
        Dictionary containing:

        probability:
            Normalized posterior [H, W].

        log_probability:
            Normalized log posterior [H, W].

        log_likelihood:
            Per-pixel log p(g_obs | x_i), shape [H, W].

        points:
            Candidate 3D points [H, W, 3].

        valid_mask:
            Boolean valid-pixel mask [H, W].

        nominal_direction:
            Nominal finger-to-candidate directions [H, W, 3].

        tangent_covariance:
            Tangent-plane covariance for valid pixels [N, 2, 2].
    """
    depth = np.asarray(depth, dtype=np.float64)
    device = str(device).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")

    if depth.ndim != 2:
        raise ValueError(
            f"depth must have shape [H, W], got {depth.shape}"
        )

    if sigma_u <= 0:
        raise ValueError("sigma_u must be positive.")

    if alpha_depth <= 0:
        raise ValueError("alpha_depth must be positive.")

    if covariance_jitter <= 0:
        raise ValueError("covariance_jitter must be positive.")

    if not np.isfinite(fitting_covariance_beta) or fitting_covariance_beta < 0:
        raise ValueError("fitting_covariance_beta must be finite and non-negative.")

    eye_position = np.asarray(
        eye_position,
        dtype=np.float64,
    )
    finger_position = np.asarray(
        finger_position,
        dtype=np.float64,
    )
    observed_direction = np.asarray(
        observed_direction,
        dtype=np.float64,
    )

    if eye_position.shape != (3,):
        raise ValueError("eye_position must have shape [3].")

    if finger_position.shape != (3,):
        raise ValueError("finger_position must have shape [3].")

    if observed_direction.shape != (3,):
        raise ValueError("observed_direction must have shape [3].")

    if valid_mask is not None and np.asarray(valid_mask).shape != depth.shape:
        raise ValueError(
            f"valid_mask must have shape {depth.shape}, got {np.asarray(valid_mask).shape}"
        )
    if log_prior is not None and np.asarray(log_prior).shape != depth.shape:
        raise ValueError(
            f"log_prior must have shape {depth.shape}, got {np.asarray(log_prior).shape}"
        )
    if camera_to_world is not None and np.asarray(camera_to_world).shape != (4, 4):
        raise ValueError(
            f"transform must have shape [4, 4], got {np.asarray(camera_to_world).shape}"
        )

    observed_direction = normalize_vectors(
        observed_direction
    )

    fitting_covariance_3d = np.zeros((3, 3), dtype=np.float64)
    if fitting_joint_coordinates is not None:
        joints = np.asarray(fitting_joint_coordinates, dtype=np.float64)
        if (joints.ndim != 2 or joints.shape[1] != 3 or len(joints) < 2
                or not np.all(np.isfinite(joints))):
            raise ValueError(
                "fitting_joint_coordinates must contain at least two finite 3D points."
            )
        centered_joints = joints - joints.mean(axis=0, keepdims=True)
        eigenvalues, eigenvectors = np.linalg.eigh(centered_joints.T @ centered_joints)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        if eigenvalues[0] <= 1e-12:
            raise ValueError("The fitting joints must not all coincide.")

        fitted_direction = eigenvectors[:, 0]
        if np.dot(fitted_direction, observed_direction) < 0:
            fitted_direction = -fitted_direction
        observed_direction = fitted_direction

        v2 = eigenvectors[:, 1]
        v3 = eigenvectors[:, 2]
        fitting_covariance_3d = fitting_covariance_beta * (
            (eigenvalues[1] / eigenvalues[0]) * np.outer(v2, v2)
            + (eigenvalues[2] / eigenvalues[0]) * np.outer(v3, v3)
        )

    if device != "cpu":
        from .gesture_torch import gaussian_posterior, resolve_device
        torch, resolved_device = resolve_device(device)
        if resolved_device is not None:
            return gaussian_posterior(
                torch, resolved_device, depth, intrinsics, eye_position,
                finger_position, observed_direction, sigma_u, alpha_depth,
                camera_to_world, valid_mask, log_prior, min_depth, max_depth,
                covariance_jitter, fitting_covariance_3d,
            )

    height, width = depth.shape

    # ----------------------------------------------------------
    # 1. Back-project all depth pixels to 3D.
    # ----------------------------------------------------------
    points = depth_to_points(
        depth=depth,
        intrinsics=intrinsics,
    )

    if camera_to_world is not None:
        points = transform_points(
            points=points,
            transform=camera_to_world,
        )

    # ----------------------------------------------------------
    # 2. Determine valid candidate pixels.
    # ----------------------------------------------------------
    current_valid_mask = (
        np.isfinite(depth)
        & (depth > min_depth)
        & (depth < max_depth)
    )

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != depth.shape:
            raise ValueError(
                f"valid_mask must have shape {depth.shape}, "
                f"got {valid_mask.shape}"
            )

        current_valid_mask &= valid_mask

    flat_valid_mask = current_valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid_mask)

    if valid_indices.size == 0:
        raise ValueError("No valid depth pixels were found.")

    flat_points = points.reshape(-1, 3)
    candidate_points = flat_points[valid_indices]

    number_of_candidates = candidate_points.shape[0]

    # ----------------------------------------------------------
    # 3. Candidate eye direction u_i and depth d_i.
    # ----------------------------------------------------------
    eye_to_candidate = (
        candidate_points - eye_position[None, :]
    )

    d = np.linalg.norm(
        eye_to_candidate,
        axis=1,
    )
    d = np.maximum(d, 1e-10)

    u = eye_to_candidate / d[:, None]

    # ----------------------------------------------------------
    # 4. Nominal gesture direction mu_i.
    # ----------------------------------------------------------
    finger_to_candidate = (
        candidate_points - finger_position[None, :]
    )

    r = np.linalg.norm(
        finger_to_candidate,
        axis=1,
    )
    r = np.maximum(r, 1e-10)

    mu = finger_to_candidate / r[:, None]

    # ----------------------------------------------------------
    # 5. Construct tangent-plane bases.
    # ----------------------------------------------------------
    b1, b2 = make_tangent_basis(mu)

    # Shape: [N, 3, 2]
    tangent_basis = np.stack(
        [b1, b2],
        axis=-1,
    )

    # ----------------------------------------------------------
    # 6. Construct Sigma_y for each candidate point.
    # ----------------------------------------------------------
    identity_3 = np.eye(3, dtype=np.float64)[None, :, :]

    uu_t = (
        u[:, :, None] * u[:, None, :]
    )

    sigma_d = alpha_depth * d

    depth_covariance = (
        sigma_d**2
    )[:, None, None] * uu_t

    angular_covariance = (
        d**2 * sigma_u**2
    )[:, None, None] * (identity_3 - uu_t)

    sigma_y = (
        depth_covariance
        + angular_covariance
    )

    # ----------------------------------------------------------
    # 7. Project Sigma_y into the tangent plane of mu_i.
    #
    # Since columns of B are already orthogonal to mu:
    #
    #   B^T P Sigma_y P B = B^T Sigma_y B
    #
    # Therefore:
    #
    #   C_i = B_i^T Sigma_y_i B_i / r_i^2
    # ----------------------------------------------------------
    tangent_covariance = np.einsum(
        "nai,nab,nbj->nij",
        tangent_basis,
        sigma_y,
        tangent_basis,
    )

    tangent_covariance /= (
        r**2
    )[:, None, None]

    fitting_tangent_covariance = np.einsum(
        "nai,ab,nbj->nij",
        tangent_basis,
        fitting_covariance_3d,
        tangent_basis,
    )
    tangent_covariance += fitting_tangent_covariance

    tangent_covariance += (
        covariance_jitter
        * np.eye(2, dtype=np.float64)[None, :, :]
    )

    # ----------------------------------------------------------
    # 8. Map the observed direction to each tangent plane.
    # ----------------------------------------------------------
    tangent_error_3d = sphere_log_map(
        mu=mu,
        g=observed_direction,
    )

    # z_i = B_i^T Log_mu_i(g_obs)
    tangent_error_2d = np.einsum(
        "nai,na->ni",
        tangent_basis,
        tangent_error_3d,
    )

    # ----------------------------------------------------------
    # 9. Evaluate the two-dimensional Gaussian likelihood.
    # ----------------------------------------------------------
    covariance_a = tangent_covariance[:, 0, 0]
    covariance_b = tangent_covariance[:, 0, 1]
    covariance_c = tangent_covariance[:, 1, 1]

    determinant = (
        covariance_a * covariance_c
        - covariance_b**2
    )

    valid_covariance = determinant > 0

    determinant_safe = np.maximum(
        determinant,
        covariance_jitter**2,
    )

    error_1 = tangent_error_2d[:, 0]
    error_2 = tangent_error_2d[:, 1]

    mahalanobis_distance = (
        covariance_c * error_1**2
        - 2.0 * covariance_b * error_1 * error_2
        + covariance_a * error_2**2
    ) / determinant_safe

    valid_log_likelihood = (
        -np.log(2.0 * np.pi)
        - 0.5 * np.log(determinant_safe)
        - 0.5 * mahalanobis_distance
    )

    valid_log_likelihood[~valid_covariance] = -np.inf

    # ----------------------------------------------------------
    # 10. Add the prior.
    # ----------------------------------------------------------
    if log_prior is None:
        valid_log_prior = np.zeros(
            number_of_candidates,
            dtype=np.float64,
        )
    else:
        log_prior = np.asarray(
            log_prior,
            dtype=np.float64,
        )

        if log_prior.shape != depth.shape:
            raise ValueError(
                f"log_prior must have shape {depth.shape}, "
                f"got {log_prior.shape}"
            )

        valid_log_prior = (
            log_prior.reshape(-1)[valid_indices]
        )

    unnormalized_log_posterior = (
        valid_log_likelihood
        + valid_log_prior
    )

    # ----------------------------------------------------------
    # 11. Normalize across all valid pixels.
    # ----------------------------------------------------------
    normalization_constant = logsumexp(
        unnormalized_log_posterior
    )

    if not np.isfinite(normalization_constant):
        raise FloatingPointError(
            "Posterior normalization failed. "
            "Check uncertainty parameters and coordinate frames."
        )

    valid_log_probability = (
        unnormalized_log_posterior
        - normalization_constant
    )

    valid_probability = np.exp(
        valid_log_probability
    )

    # ----------------------------------------------------------
    # 12. Scatter values back to image coordinates.
    # ----------------------------------------------------------
    flat_probability = np.zeros(
        height * width,
        dtype=np.float64,
    )
    flat_probability[valid_indices] = valid_probability

    flat_log_probability = np.full(
        height * width,
        -np.inf,
        dtype=np.float64,
    )
    flat_log_probability[
        valid_indices
    ] = valid_log_probability

    flat_log_likelihood = np.full(
        height * width,
        -np.inf,
        dtype=np.float64,
    )
    flat_log_likelihood[
        valid_indices
    ] = valid_log_likelihood

    flat_nominal_direction = np.zeros(
        (height * width, 3),
        dtype=np.float64,
    )
    flat_nominal_direction[
        valid_indices
    ] = mu

    return {
        "probability": flat_probability.reshape(
            height,
            width,
        ),
        "log_probability": flat_log_probability.reshape(
            height,
            width,
        ),
        "log_likelihood": flat_log_likelihood.reshape(
            height,
            width,
        ),
        "points": points,
        "valid_mask": current_valid_mask,
        "nominal_direction": flat_nominal_direction.reshape(
            height,
            width,
            3,
        ),
        "tangent_covariance": tangent_covariance,
        "fitting_covariance": fitting_covariance_3d,
        "valid_indices": valid_indices,
    }


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    fx = fy = cx = cy = None
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("fx:"):
                fx = float(line.split(":", 1)[1].strip())
            elif line.startswith("fy:"):
                fy = float(line.split(":", 1)[1].strip())
            elif line.startswith("cx:"):
                cx = float(line.split(":", 1)[1].strip())
            elif line.startswith("cy:"):
                cy = float(line.split(":", 1)[1].strip())

    if None in (fx, fy, cx, cy):
        raise RuntimeError(
            f"Failed to parse fx/fy/cx/cy from {path}"
        )

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def load_depth_meters(path: Path, depth_scale: float = 0.001) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")

    return depth_raw.astype(np.float64) * depth_scale


def load_ray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as file:
        ray = json.load(file)

    finger_position = np.asarray(ray["origin"], dtype=np.float64)
    observed_direction = np.asarray(ray["direction"], dtype=np.float64)
    return finger_position, observed_direction


def load_fitting_joint_coordinates(path: Path, joint_count: int = 4) -> np.ndarray:
    """Load the requested distal ordered 3D index-finger joints used by the fit."""
    if joint_count not in {3, 4}:
        raise ValueError("joint_count must be 3 or 4.")
    with open(path, "r", encoding="utf-8") as file:
        ray = json.load(file)
    fitting_points = ray.get("fitting_points_3d")
    if fitting_points is not None:
        try:
            points = np.asarray(fitting_points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid fitting_points_3d in {path}.") from error
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.all(np.isfinite(points))):
            raise ValueError(
                f"fitting_points_3d in {path} must contain at least two finite 3D points."
            )
        return points
    landmarks = ray.get("index_finger_landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 4:
        raise ValueError(
            f"{path} must contain exactly four index_finger_landmarks."
        )
    try:
        ordered = sorted(landmarks, key=lambda item: int(item["landmark_index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid index_finger_landmarks in {path}.") from error

    try:
        points = np.asarray(
            [item["point_3d"] for item in ordered[-joint_count:]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid index_finger_landmarks in {path}.") from error
    if points.shape != (joint_count, 3) or not np.all(np.isfinite(points)):
        raise ValueError(
            f"The requested {joint_count} distal index-finger point_3d values "
            f"in {path} must be finite."
        )
    return points


def compute_eye_position(
    finger_position: np.ndarray,
    ray_direction: np.ndarray,
    back_distance_m: float = 0.45,
    up_distance_m: float = 0.5,
) -> np.ndarray:
    """
    Place the eye behind the finger along the inverse ray, then move up.

    Camera frame convention: X right, Y down, Z forward.
    "Up" is therefore -Y.
    """
    ray_direction = normalize_vectors(
        np.asarray(ray_direction, dtype=np.float64)
    )
    camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    return (
        finger_position
        - back_distance_m * ray_direction
        + up_distance_m * camera_up
    )


def save_posterior_visualization(
    rgb_bgr: np.ndarray,
    probability: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    heatmap = np.zeros_like(rgb_bgr, dtype=np.float64)
    valid_probability = probability[valid_mask]

    if valid_probability.size == 0:
        normalized = probability.copy()
    else:
        maximum = valid_probability.max()
        normalized = probability / maximum if maximum > 0 else probability

    # heatmap[..., 2] = normalized * 255.0
    # heatmap[..., 0] = normalized * 40.0

    gray = normalized * 255.0
    heatmap[..., 0] = gray
    heatmap[..., 1] = gray
    heatmap[..., 2] = gray

    overlay = rgb_bgr.astype(np.float64)
    overlay[valid_mask] = (
        # 0.1 * overlay[valid_mask]
        + 0.9 * heatmap[valid_mask]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def run_frame_00001(
    root: Optional[Path] = None,
    frame_id: str = "00001",
    sigma_u: float = 0.009,
    alpha_depth: float = 0.2,
) -> dict[str, np.ndarray]:
    root = Path(root) if root is not None else Path(__file__).parent

    depth_path = root / "key_depth" / f"{frame_id}.png"
    rgb_path = root / "key_rgb" / f"{frame_id}.png"
    ray_path = root / "key_frames" / f"{frame_id}.json"
    intrinsics_path = root / "camera_intrinsics.txt"

    depth = load_depth_meters(depth_path)
    intrinsics = load_camera_intrinsics(intrinsics_path)
    finger_position, observed_direction = load_ray(ray_path)
    eye_position = compute_eye_position(
        finger_position=finger_position,
        ray_direction=observed_direction,
    )

    result = pixel_posterior_from_gesture(
        depth=depth,
        intrinsics=intrinsics,
        eye_position=eye_position,
        finger_position=finger_position,
        observed_direction=observed_direction,
        sigma_u=sigma_u,
        alpha_depth=alpha_depth,
    )

    output_dir = root / "outputs" / "gesture_posterior"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / f"{frame_id}_posterior.npz",
        probability=result["probability"],
        log_probability=result["log_probability"],
        log_likelihood=result["log_likelihood"],
        valid_mask=result["valid_mask"],
        eye_position=eye_position,
        finger_position=finger_position,
        observed_direction=normalize_vectors(observed_direction),
    )

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is not None:
        save_posterior_visualization(
            rgb_bgr=rgb_bgr,
            probability=result["probability"],
            valid_mask=result["valid_mask"],
            output_path=output_dir / f"{frame_id}_posterior_overlay.png",
        )

    valid_probability = result["probability"][result["valid_mask"]]
    peak_flat_index = int(np.argmax(result["probability"]))
    peak_v, peak_u = np.unravel_index(
        peak_flat_index,
        result["probability"].shape,
    )

    print(f"Frame: {frame_id}")
    print(f"Valid pixels: {result['valid_mask'].sum()}")
    print(f"Eye position: {eye_position}")
    print(f"Finger position: {finger_position}")
    print(
        "Observed direction: "
        f"{normalize_vectors(observed_direction)}"
    )
    print(f"Peak probability: {valid_probability.max():.6e}")
    print(f"Peak pixel (v, u): ({peak_v}, {peak_u})")
    print(f"Saved results to: {output_dir}")

    return result


if __name__ == "__main__":
    run_frame_00001()
