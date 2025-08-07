import math

def prog_disc(obs: float, proj: float, scale: float, alpha: float) -> float:
    """
    Progressively discount extreme deviations from expected value.

    Args:
        obs (float): Observed value.
        proj (float): Projected/expected value.
        scale (float): Normalization scale.
        alpha (float): Discounting strength (0.001–0.005 typical).

    Returns:
        float: Discounted observed value closer to projection.
    """
    abs_error = abs(obs - proj)
    if abs_error == 0 or alpha == 0:
        return obs

    error_dir = 1 if obs >= proj else -1
    max_adj = 0.309 * (alpha ** -0.864) * scale
    scaled_error = min(abs_error, max_adj)
    dampen_factor = (scaled_error / scale) * alpha
    dampen_factor = min(dampen_factor, 1)

    try:
        adjusted = proj + error_dir * (scaled_error ** (1 - dampen_factor))
        return adjusted
    except OverflowError:
        return obs