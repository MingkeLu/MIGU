"""Optional PyTorch/CUDA kernels for pixel-wise gesture posteriors."""

from __future__ import annotations

import numpy as np


def resolve_device(device: str):
    """Import torch lazily and resolve ``auto`` without making it mandatory."""
    requested = str(device).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    try:
        import torch
    except ImportError as exc:
        if requested == "auto":
            return None, None
        raise RuntimeError(
            "The PyTorch backend was requested but torch is not installed."
        ) from exc
    if requested == "auto":
        if not torch.cuda.is_available():
            return None, None
        requested = "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch, torch.device(requested)


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _points(torch, depth, intrinsics):
    height, width = depth.shape
    v = torch.arange(height, dtype=depth.dtype, device=depth.device)[:, None]
    u = torch.arange(width, dtype=depth.dtype, device=depth.device)[None, :]
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    return torch.stack((x.expand_as(depth), y.expand_as(depth), depth), dim=-1)


def vmf_posterior(
    torch, device, depth, intrinsics, origin, axis, kappa, valid_mask,
    min_depth, max_depth,
):
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    depth_t = torch.as_tensor(depth, dtype=dtype, device=device)
    origin_t = torch.as_tensor(origin, dtype=dtype, device=device)
    axis_t = torch.as_tensor(axis, dtype=dtype, device=device)
    axis_t = axis_t / torch.linalg.vector_norm(axis_t).clamp_min(1e-12)
    points = _points(torch, depth_t, intrinsics)
    offsets = points - origin_t
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    valid = (torch.isfinite(depth_t) & (depth_t > min_depth) & (depth_t < max_depth)
             & torch.isfinite(points).all(dim=-1) & (distances > 1e-12))
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    if not bool(valid.any().item()):
        raise ValueError("No valid depth pixels are available for the vMF posterior.")
    directions = offsets / distances[..., None].clamp_min(1e-12)
    log_likelihood = torch.full_like(depth_t, -torch.inf)
    log_likelihood[valid] = kappa * (directions[valid] * axis_t).sum(dim=-1)
    log_probability = torch.full_like(depth_t, -torch.inf)
    log_probability[valid] = log_likelihood[valid] - torch.logsumexp(log_likelihood[valid], 0)
    probability = torch.zeros_like(depth_t)
    probability[valid] = torch.exp(log_probability[valid])
    return {
        "probability": _numpy(probability).astype(np.float64),
        "log_probability": _numpy(log_probability).astype(np.float64),
        "log_likelihood": _numpy(log_likelihood).astype(np.float64),
        "valid_mask": _numpy(valid),
        "points": _numpy(points).astype(np.float64),
        "nominal_direction": _numpy(directions).astype(np.float64),
        "fitting_covariance": np.zeros((3, 3), dtype=np.float64),
    }


def gaussian_posterior(
    torch, device, depth, intrinsics, eye, finger, observed, sigma_u,
    alpha_depth, camera_to_world, valid_mask, log_prior, min_depth, max_depth,
    jitter, fitting_covariance,
):
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    depth_t = torch.as_tensor(depth, dtype=dtype, device=device)
    eye = torch.as_tensor(eye, dtype=dtype, device=device)
    finger = torch.as_tensor(finger, dtype=dtype, device=device)
    observed = torch.as_tensor(observed, dtype=dtype, device=device)
    observed = observed / torch.linalg.vector_norm(observed).clamp_min(1e-12)
    fit_cov = torch.as_tensor(fitting_covariance, dtype=dtype, device=device)
    points = _points(torch, depth_t, intrinsics)
    if camera_to_world is not None:
        transform = torch.as_tensor(camera_to_world, dtype=dtype, device=device)
        points = torch.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]
    valid = torch.isfinite(depth_t) & (depth_t > min_depth) & (depth_t < max_depth)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    indices = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    if indices.numel() == 0:
        raise ValueError("No valid depth pixels were found.")
    candidates = points.reshape(-1, 3)[indices]
    eye_delta = candidates - eye
    d = torch.linalg.vector_norm(eye_delta, dim=1).clamp_min(1e-10)
    u = eye_delta / d[:, None]
    finger_delta = candidates - finger
    r = torch.linalg.vector_norm(finger_delta, dim=1).clamp_min(1e-10)
    mu = finger_delta / r[:, None]

    reference = torch.zeros_like(mu)
    use_z = mu[:, 2].abs() < .9
    reference[use_z, 2] = 1
    reference[~use_z, 1] = 1
    b1 = torch.linalg.cross(mu, reference)
    b1 /= torch.linalg.vector_norm(b1, dim=1, keepdim=True).clamp_min(1e-12)
    b2 = torch.linalg.cross(mu, b1)
    b2 /= torch.linalg.vector_norm(b2, dim=1, keepdim=True).clamp_min(1e-12)
    basis = torch.stack((b1, b2), dim=-1)

    identity3 = torch.eye(3, dtype=dtype, device=device)[None]
    uu = u[:, :, None] * u[:, None, :]
    sigma_y = ((alpha_depth * d) ** 2)[:, None, None] * uu
    sigma_y += (d ** 2 * sigma_u ** 2)[:, None, None] * (identity3 - uu)
    covariance = torch.einsum("nai,nab,nbj->nij", basis, sigma_y, basis)
    covariance /= (r ** 2)[:, None, None]
    covariance += torch.einsum("nai,ab,nbj->nij", basis, fit_cov, basis)
    covariance += jitter * torch.eye(2, dtype=dtype, device=device)[None]

    cosine = (mu * observed[None]).sum(1).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cosine)
    scale = torch.where(theta >= 1e-5, theta / torch.sin(theta).clamp_min(1e-10), 1.)
    tangent_error = scale[:, None] * (observed[None] - cosine[:, None] * mu)
    error = torch.einsum("nai,na->ni", basis, tangent_error)
    a, b, c = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    determinant = a * c - b.square()
    determinant_safe = determinant.clamp_min(jitter ** 2)
    mahalanobis = (c * error[:, 0].square() - 2 * b * error[:, 0] * error[:, 1]
                   + a * error[:, 1].square()) / determinant_safe
    likelihood = -np.log(2 * np.pi) - .5 * torch.log(determinant_safe) - .5 * mahalanobis
    likelihood = torch.where(determinant > 0, likelihood, -torch.inf)
    if log_prior is None:
        prior = torch.zeros_like(likelihood)
    else:
        prior = torch.as_tensor(log_prior, dtype=dtype, device=device).reshape(-1)[indices]
    posterior = likelihood + prior
    normalizer = torch.logsumexp(posterior, 0)
    if not bool(torch.isfinite(normalizer).item()):
        raise FloatingPointError("Posterior normalization failed. Check uncertainty parameters and coordinate frames.")
    log_probability_valid = posterior - normalizer
    size = depth_t.numel()
    probability = torch.zeros(size, dtype=dtype, device=device)
    log_probability = torch.full((size,), -torch.inf, dtype=dtype, device=device)
    log_likelihood = torch.full((size,), -torch.inf, dtype=dtype, device=device)
    nominal = torch.zeros((size, 3), dtype=dtype, device=device)
    probability[indices] = torch.exp(log_probability_valid)
    log_probability[indices] = log_probability_valid
    log_likelihood[indices] = likelihood
    nominal[indices] = mu
    height, width = depth_t.shape
    return {
        "probability": _numpy(probability.reshape(height, width)).astype(np.float64),
        "log_probability": _numpy(log_probability.reshape(height, width)).astype(np.float64),
        "log_likelihood": _numpy(log_likelihood.reshape(height, width)).astype(np.float64),
        "points": _numpy(points).astype(np.float64), "valid_mask": _numpy(valid),
        "nominal_direction": _numpy(nominal.reshape(height, width, 3)).astype(np.float64),
        "tangent_covariance": _numpy(covariance).astype(np.float64),
        "fitting_covariance": np.asarray(fitting_covariance, dtype=np.float64),
        "valid_indices": _numpy(indices),
    }
