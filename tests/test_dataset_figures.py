from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset_figures import (
    add_hydrological_year,
    generate_dataset_figures,
    load_processed_dataset,
    prepare_hydrological_monthly_storage,
    prepare_monthly_total_storage,
    prepare_seasonal_envelope,
    standardize_dataset_columns,
)


def build_synthetic_daily_dataset() -> pd.DataFrame:
    dates = pd.date_range("2013-12-01", "2015-11-30", freq="D")
    n = len(dates)

    data = pd.DataFrame(
        {
            "FECHA": dates,
            "VOL_TOTAL_HM3": [93.0] * n,
            "VOL_TOTAL_REL": [0.25] * n,
            "AH_YEAR": [2014 if date.month == 12 or date.year == 2014 else 2015 for date in dates],
            "AH": [
                f"AH-{2014 if date.month == 12 or date.year == 2014 else 2015}"
                for date in dates
            ],
        }
    )

    return data


def test_standardize_dataset_columns_maps_real_pipeline_columns():
    data = pd.DataFrame(
        {
            "FECHA": ["2011-12-01"],
            "VOL_TOTAL_HM3": [204.8],
            "VOL_TOTAL_REL": [0.44],
        }
    )

    result = standardize_dataset_columns(data)

    assert "fecha" in result.columns
    assert "vol_total_hm3" in result.columns
    assert "vol_total_rel" in result.columns


def test_load_processed_dataset_accepts_real_pipeline_column_names(tmp_path: Path):
    data = build_synthetic_daily_dataset()
    dataset_path = tmp_path / "dataset_diario.csv"
    data.to_csv(dataset_path, index=False)

    result = load_processed_dataset(dataset_path)

    assert "fecha" in result.columns
    assert "vol_total_hm3" in result.columns
    assert "vol_total_rel" in result.columns


def test_add_hydrological_year_assigns_december_to_next_year():
    data = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                ["2013-11-30", "2013-12-01", "2014-01-01"]
            )
        }
    )

    result = add_hydrological_year(data)

    assert result["ah_year"].tolist() == [2013, 2014, 2014]
    assert result["ah"].tolist() == ["AH-2013", "AH-2014", "AH-2014"]


def test_prepare_monthly_total_storage_returns_24_months_for_two_years():
    data = build_synthetic_daily_dataset()

    monthly = prepare_monthly_total_storage(data)

    assert len(monthly) == 24
    assert "monthly_mean_hm3" in monthly.columns
    assert monthly["monthly_mean_hm3"].iloc[0] == 93.0


def test_prepare_hydrological_monthly_storage_returns_complete_hydrological_grid():
    data = build_synthetic_daily_dataset()

    hydro_monthly = prepare_hydrological_monthly_storage(data)

    assert "AH_YEAR" in hydro_monthly.columns
    assert "hydro_month" in hydro_monthly.columns
    assert "normalized_storage" in hydro_monthly.columns
    assert hydro_monthly["AH_YEAR"].nunique() == 2
    assert hydro_monthly["hydro_month"].nunique() == 12


def test_prepare_seasonal_envelope_returns_12_hydrological_months():
    data = build_synthetic_daily_dataset()
    hydro_monthly = prepare_hydrological_monthly_storage(data)

    envelope = prepare_seasonal_envelope(hydro_monthly)

    assert len(envelope) == 12
    assert set(
        ["min_value", "q25", "median_value", "q75", "max_value"]
    ).issubset(envelope.columns)


def test_generate_dataset_figures_creates_two_png_files(tmp_path: Path):
    data = build_synthetic_daily_dataset()
    dataset_path = tmp_path / "dataset_diario.csv"
    data.to_csv(dataset_path, index=False)

    output_dir = tmp_path / "figures"

    pipeline_config = {
        "embalses": {"capacidad_util_maxima_sistema_hm3": 393.0},
        "rutas": {
            "dataset_procesado": str(dataset_path),
            "carpeta_figuras": str(output_dir),
        }
    }

    result = generate_dataset_figures(pipeline_config)

    assert "figure_1" in result
    assert "figure_2" in result
    assert result["figure_1"].exists()
    assert result["figure_2"].exists()
    assert result["figure_1"].suffix == ".png"
    assert result["figure_2"].suffix == ".png"
