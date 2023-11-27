from typing import Union

import numpy as np


def apply_fried_height_adjustment(cn2: Union['pd.DataFrame', 'pd.Series', np.ndarray], observed: float,
                                  desired: float) -> Union['pd.DataFrame', 'pd.Series', np.ndarray]:
    """Height adjustment, see Chen 2019 Astron. Astrophys. 19 015.

    Args:
        cn2 (pd.Series): The Cn2 values to adjust.
        desired (float): The desired height.
        observed (float, optional): The observed height. Defaults to 0.0.
    
    Returns:
        pd.Series: The adjusted Cn2 values.
    """
    try:
        assert observed >= 0.0
        assert desired > 0.0
    except AssertionError:
        raise ValueError('The observed and desired heights must be positive.')

    if observed > 0:
        cn2 /= ((observed**(-1 / 3)) * np.exp(-1 * observed / 3200))
        cn2 *= ((desired**(-1 / 3)) * np.exp(-1 * desired / 3200))
    return cn2


def apply_oermann_height_adjustment(
        cn2: Union['pd.DataFrame', 'pd.Series', np.ndarray],
        desired: float,
        observed: float,
        power_law_scaling: float = -4 / 3) -> Union['pd.DataFrame', 'pd.Series', np.ndarray]:
    """Height adjustment, see Oermann 2014, use -2/3 or -4/3 scaling laws.
    
    Args:
        cn2 (pd.Series): The Cn2 values to adjust.
        desired (float): The desired height.
        observed (float, optional): The observed height. Defaults to 15.0.
        power_law_scaling (float, optional): The power law scaling. Defaults to -4/3.

    Returns:
        pd.Series: The adjusted Cn2 values.
    """
    try:
        assert observed > 0.0
        assert desired > 0.0
    except AssertionError:
        raise ValueError('The observed and desired heights must be positive.')

    cn2 *= (desired / observed)**(power_law_scaling)

    return cn2
