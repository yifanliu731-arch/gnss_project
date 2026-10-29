"""CSV profile loader shared by the field-upload API."""
import pandas as pd


def load_ssp_from_csv(csv_path) -> pd.DataFrame:
    """Load SSP CSV (path or file-like) and normalise columns for RayTracer."""
    from ray_tracing import normalize_ssp_columns
    df = pd.read_csv(csv_path)
    return normalize_ssp_columns(df)[['Depth (m)', 'SV (m/s)']].dropna()
