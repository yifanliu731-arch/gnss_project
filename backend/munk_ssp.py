"""
Munk Sound-Speed Profile (SSP) model for SeaArrayCalib.

Provides a canonical deep-water SSP with a well-defined SOFAR channel axis,
widely used in underwater acoustics for simulation and validation.

Model
-----
The idealised Munk profile  [Munk, W.H., JASA 55(2), 1974]:

    C(z)  =  c_axis × [ 1  +  ε × ( η − 1  +  e^{−η} ) ]

where

    η     =  2 × (z − z_axis) / B                    (dimensionless depth)
    z       =  depth (positive downward, m)
    z_axis  =  SOFAR channel axis depth (m),          default 1000
    c_axis  =  sound speed at the axis (m/s),         default 1500
    ε       =  perturbation strength (dimensionless), default 0.00737
    B       =  thermocline thickness scale (m),       default 1300

The first derivative (vertical gradient) for validation:

    dC/dz = c_axis × ε × (2 / B) × ( 1 − e^{−η} )

Public API
----------
-  get_munk_svp(z, ...)           → float or np.ndarray
-  generate_munk_ssp_table(H, dz) → pd.DataFrame  (Depth (m), SV (m/s))
-  save_munk_ssp_csv(H, dz, path) → str           (path written)
"""

import numpy as np
import pandas as pd
import os
from typing import Union


def get_munk_svp(
    z: Union[float, np.ndarray],
    z_axis: float = 1000.0,
    c_axis: float = 1500.0,
    epsilon: float = 0.00737,
    B: float = 1300.0,
) -> Union[float, np.ndarray]:
    """
    Idealised Munk sound-speed profile.

    .. math::

        C(z) = c_{axis} \\times \\bigl[
            1 + \\varepsilon \\times
            (\\eta - 1 + e^{-\\eta})
        \\bigr]

        \\eta = \\frac{2 \\, (z - z_{axis})}{B}

    Parameters
    ----------
    z : float or np.ndarray
        Depth (m), positive downward.
    z_axis : float
        SOFAR channel axis depth (m).  Default 1000.
    c_axis : float
        Minimum sound speed at the axis (m/s).  Default 1500.
    epsilon : float
        Perturbation strength.  Default 0.00737 (~ 1% variation).
    B : float
        Thermocline thickness scale (m).  Default 1300.

    Returns
    -------
    c : float or np.ndarray
        Sound speed at each depth (m/s).

    Notes
    -----
    The profile is symmetric about z_axis: above the axis sound speed
    decreases with depth (negative gradient), below the axis it increases
    (positive gradient).  This creates the classic SOFAR channel.
    """
    eta = 2.0 * (np.asarray(z) - z_axis) / B
    c = c_axis * (1.0 + epsilon * (eta - 1.0 + np.exp(-eta)))
    return c


def _munk_gradient(
    z: Union[float, np.ndarray],
    z_axis: float = 1000.0,
    c_axis: float = 1500.0,
    epsilon: float = 0.00737,
    B: float = 1300.0,
) -> Union[float, np.ndarray]:
    """
    First derivative of the Munk profile w.r.t. depth.

    .. math::

        \\frac{dC}{dz} = c_{axis} × ε × \\frac{2}{B} × (1 − e^{−η})
    """
    eta = 2.0 * (np.asarray(z) - z_axis) / B
    grad = c_axis * epsilon * (2.0 / B) * (1.0 - np.exp(-eta))
    return grad


def generate_munk_ssp_table(
    H: float,
    dz: float = 10.0,
    z_axis: float = 1000.0,
    c_axis: float = 1500.0,
    epsilon: float = 0.00737,
    B: float = 1300.0,
) -> pd.DataFrame:
    """
    Generate a discrete Sound-Speed Profile table from the Munk model.

    The table runs from depth 0 to ``H`` with step ``dz``.

    Parameters
    ----------
    H : float
        Maximum depth (m).  The table runs from 0 to H.
    dz : float
        Depth step (m).  Default 10.
    z_axis, c_axis, epsilon, B :
        See ``get_munk_svp``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Depth (m)``, ``SV (m/s)``.
    """
    depths = np.arange(0.0, H + dz / 2, dz)
    speeds = get_munk_svp(depths, z_axis=z_axis, c_axis=c_axis,
                          epsilon=epsilon, B=B)
    # Also include surface point at exactly 0 m if not already present
    if depths[0] > 1e-6:
        depths = np.insert(depths, 0, 0.0)
        speeds = np.insert(speeds, 0, get_munk_svp(0.0, z_axis=z_axis,
                         c_axis=c_axis, epsilon=epsilon, B=B))

    df = pd.DataFrame({
        "Depth (m)": np.round(depths, 2),
        "SV (m/s)": np.round(speeds, 4),
    })
    return df


def save_munk_ssp_csv(
    H: float,
    dz: float = 10.0,
    output_path: str = "data/munk_ssp.csv",
    z_axis: float = 1000.0,
    c_axis: float = 1500.0,
    epsilon: float = 0.00737,
    B: float = 1300.0,
) -> str:
    """
    Generate a Munk SSP table and save it as a CSV file.

    Parameters
    ----------
    H, dz, z_axis, c_axis, epsilon, B :
        See ``generate_munk_ssp_table``.
    output_path : str
        Output CSV file path (relative or absolute).

    Returns
    -------
    str
        Absolute path to the written file.
    """
    df = generate_munk_ssp_table(H, dz, z_axis=z_axis, c_axis=c_axis,
                                 epsilon=epsilon, B=B)
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"[MUNK] Saved SSP table with {len(df)} layers to {abs_path}")
    print(f"       Depth range: {df['Depth (m)'].iloc[0]:.1f} – "
          f"{df['Depth (m)'].iloc[-1]:.1f} m")
    print(f"       SV min: {df['SV (m/s)'].min():.2f} m/s  "
          f"at z = {df.loc[df['SV (m/s)'].idxmin(), 'Depth (m)']:.1f} m")
    return abs_path


# =====================================================================
#  Smoke test
# =====================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # ----------------------------------------------------------------
    #  1. Plot the Munk profile in a typical deep-water scenario
    # ----------------------------------------------------------------
    H_test = 4000.0
    zz = np.linspace(0, H_test, 1000)
    cc = get_munk_svp(zz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(cc, zz, "b-", linewidth=2)
    ax1.invert_yaxis()
    ax1.set_xlabel("Sound Speed (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Idealised Munk SSP")
    ax1.grid(True, alpha=0.3)

    # Mark the SOFAR axis
    c_at_axis = get_munk_svp(1000.0)
    ax1.axhline(1000, color="red", linestyle="--", alpha=0.5,
                label=f"Axis at z={1000} m\nc_min = {c_at_axis:.2f} m/s")
    ax1.legend(fontsize=8)

    # Gradient plot
    grad = _munk_gradient(zz)
    ax2.plot(grad, zz, "g-", linewidth=2)
    ax2.invert_yaxis()
    ax2.axvline(0, color="grey", linestyle="--")
    ax2.set_xlabel(r"d$C$/d$z$ (s⁻¹)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Vertical Sound-Speed Gradient")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/munk_ssp_profile.png", dpi=150)
    print("Saved profile plot to data/munk_ssp_profile.png")

    # ----------------------------------------------------------------
    #  2. Generate and save a discrete table
    # ----------------------------------------------------------------
    df = generate_munk_ssp_table(H_test, dz=10.0)
    print(f"\nDiscrete table: {len(df)} layers")
    print(df.head(10))
    print("...")
    print(df.tail(5))

    out = save_munk_ssp_csv(H_test, dz=10.0, output_path="data/munk_ssp.csv")
    print(f"\nWrote {out}")

    plt.show()