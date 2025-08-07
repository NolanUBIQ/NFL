def s_curve(height: float, mp: float, x: float, direction: str = 'down') -> float:
    """
    Standard S-curve transformation used for discounting and scaling.

    Args:
        height (float): The height of the curve.
        mp (float): Midpoint for inflection.
        x (float): The input value to scale.
        direction (str): 'down' to decrease as x increases, 'up' to increase.

    Returns:
        float: Scaled value using sigmoid-like function.
    """
    base = 1 / (1 + 1.5 ** ((-1 * (x - mp)) * (10 / mp)))
    if direction == 'down':
        return (1 - base) * height
    else:
        return base * height
