from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


HYDRO_MONTH_ORDER = {
    12: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
}

HYDRO_MONTH_LABELS = [
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
]

REFERENCE_CAPACITY_HM3 = 393.0


def get_dataset_path(pipeline_config: dict[str, Any]) -> Path:
    """Return processed daily dataset path from pipeline config."""
    return Path(pipeline_config["rutas"]["dataset_procesado"])


def get_figure_output_dir(pipeline_config: dict[str, Any]) -> Path:
    """Return figure output directory from pipeline config."""
    return Path(pipeline_config["rutas"]["carpeta_figuras"])


def standardize_dataset_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize real processed dataset columns used by dataset figures."""
    prepared = data.copy()

    rename_map = {
        "FECHA": "fecha",
        "VOL_TOTAL_HM3": "vol_total_hm3",
        "VOL_TOTAL_REL": "vol_total_rel",
        "AH_YEAR": "ah_year",
        "AH": "ah",
        "TEMPORADA": "temporada",
    }

    existing_renames = {
        old: new
        for old, new in rename_map.items()
        if old in prepared.columns
    }

    prepared = prepared.rename(columns=existing_renames)

    return prepared


def load_processed_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load processed daily dataset and validate required columns."""
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    data = pd.read_csv(path)
    data = standardize_dataset_columns(data)

    required_columns = ["fecha", "vol_total_hm3", "vol_total_rel"]
    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Processed dataset missing required columns: {missing}"
        )

    data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")

    if data["fecha"].isna().any():
        raise ValueError("Column 'fecha' contains invalid dates.")

    return data


def add_hydrological_year(data: pd.DataFrame) -> pd.DataFrame:
    """Add hydrological year label using December-November convention."""
    prepared = data.copy()

    if "ah_year" not in prepared.columns:
        prepared["ah_year"] = prepared["fecha"].dt.year
        prepared.loc[prepared["fecha"].dt.month == 12, "ah_year"] += 1

    if "ah" not in prepared.columns:
        prepared["ah"] = "AH-" + prepared["ah_year"].astype(str)

    return prepared


def prepare_monthly_total_storage(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily total storage to monthly mean values."""
    prepared = standardize_dataset_columns(data)
    prepared["fecha"] = pd.to_datetime(prepared["fecha"])
    prepared["month_start"] = prepared["fecha"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        prepared.groupby("month_start", as_index=False)["vol_total_hm3"]
        .mean()
        .rename(columns={"vol_total_hm3": "monthly_mean_hm3"})
    )

    return monthly


def prepare_hydrological_monthly_storage(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily normalized storage by hydrological year and month."""
    prepared = standardize_dataset_columns(data)
    prepared["fecha"] = pd.to_datetime(prepared["fecha"])
    prepared = add_hydrological_year(prepared)

    prepared["hydro_month"] = prepared["fecha"].dt.month.map(HYDRO_MONTH_ORDER)
    prepared["month_start"] = prepared["fecha"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        prepared.groupby(
            ["ah_year", "ah", "hydro_month", "month_start"],
            as_index=False,
        )
        .agg(
            monthly_mean_hm3=("vol_total_hm3", "mean"),
            normalized_storage=("vol_total_rel", "mean"),
        )
        .rename(columns={"ah_year": "AH_YEAR", "ah": "AH"})
    )

    return monthly


def prepare_seasonal_envelope(hydro_monthly: pd.DataFrame) -> pd.DataFrame:
    """Create monthly seasonal summary for normalized storage."""
    required_columns = {"hydro_month", "normalized_storage"}
    missing = required_columns.difference(hydro_monthly.columns)

    if missing:
        raise ValueError(
            f"Hydrological monthly data missing columns: {sorted(missing)}"
        )

    envelope = (
        hydro_monthly.groupby("hydro_month")["normalized_storage"]
        .agg(
            min_value="min",
            q25=lambda x: x.quantile(0.25),
            median_value="median",
            q75=lambda x: x.quantile(0.75),
            max_value="max",
        )
        .reset_index()
        .sort_values("hydro_month")
        .reset_index(drop=True)
    )

    return envelope

def plot_figure_1_monthly_total_storage(
    monthly: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot Figure 1: monthly evolution of total useful storage."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        monthly["month_start"],
        monthly["monthly_mean_hm3"],
        linewidth=1.4,
        color="#1f4e79",
        label="Monthly mean storage",
    )

    ax.axhline(
        REFERENCE_CAPACITY_HM3,
        linestyle="--",
        linewidth=0.9,
        color="#6c757d",
        label="Maximum useful capacity",
    )

    ax.axvspan(
        pd.Timestamp("2013-12-01"),
        pd.Timestamp("2015-11-30"),
        color="#c0392b",
        alpha=0.15,
        label="Drought period: AH-2014/2015",
    )

    ax.axvspan(
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-11-30"),
        color="#2e86c1",
        alpha=0.15,
        label="Wet year: AH-2025",
    )

    december_starts = monthly.loc[
        monthly["month_start"].dt.month == 12,
        "month_start",
    ]

    for december in december_starts:
        ax.axvline(
            december,
            linestyle=":",
            linewidth=0.6,
            color="#b0b0b0",
        )

    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Monthly mean useful storage (Hm³)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.35)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=7,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output

def plot_figure_2_seasonal_storage_pattern(
    hydro_monthly: pd.DataFrame,
    envelope: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot Figure 2: seasonal pattern of normalized total useful storage."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.axvspan(
        0.5,
        6.5,
        color="#d6eaf8",
        alpha=0.45,
        label="Wet season",
    )
    ax.axvspan(
        6.5,
        12.5,
        color="#f9e79f",
        alpha=0.25,
        label="Dry season",
    )

    x = envelope["hydro_month"]

    ax.fill_between(
        x,
        envelope["min_value"],
        envelope["max_value"],
        color="#d5d8dc",
        alpha=0.55,
        label="Min-max range (14 AHs)",
    )

    ax.fill_between(
        x,
        envelope["q25"],
        envelope["q75"],
        color="#a9cce3",
        alpha=0.55,
        label="P25-P75 range",
    )

    ax.plot(
        x,
        envelope["median_value"],
        linewidth=1.8,
        marker="o",
        markersize=3,
        color="#1f4e79",
        label="Typical year (median)",
    )

    highlight_styles = {
        2014: {
            "color": "#d68910",
            "linestyle": "--",
            "label": "AH-2014",
        },
        2015: {
            "color": "#943126",
            "linestyle": "-",
            "label": "AH-2015",
        },
        2025: {
            "color": "#154360",
            "linestyle": "-",
            "label": "AH-2025",
        },
    }

    for hydrological_year, style in highlight_styles.items():
        year_data = hydro_monthly.loc[
            hydro_monthly["AH_YEAR"] == hydrological_year
        ].sort_values("hydro_month")

        if not year_data.empty:
            ax.plot(
                year_data["hydro_month"],
                year_data["normalized_storage"],
                linewidth=1.6,
                marker="s",
                markersize=3,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
            )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(HYDRO_MONTH_LABELS)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Hydrological-year month", fontsize=8)
    ax.set_ylabel("Normalized useful storage", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.35)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=7,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output

def generate_dataset_figures(
    pipeline_config: dict[str, Any],
) -> dict[str, Path]:
    """Generate the two dataset figures from processed daily data."""
    dataset_path = get_dataset_path(pipeline_config)
    output_dir = get_figure_output_dir(pipeline_config)

    data = load_processed_dataset(dataset_path)
    monthly = prepare_monthly_total_storage(data)
    hydro_monthly = prepare_hydrological_monthly_storage(data)
    envelope = prepare_seasonal_envelope(hydro_monthly)

    figure_1_path = output_dir / "figure_1_monthly_total_storage.png"
    figure_2_path = output_dir / "figure_2_seasonal_storage_pattern.png"

    plot_figure_1_monthly_total_storage(monthly, figure_1_path)
    plot_figure_2_seasonal_storage_pattern(
        hydro_monthly,
        envelope,
        figure_2_path,
    )

    print("[dataset_figures] Figures generated successfully")
    print(f"[dataset_figures] Figure 1: {figure_1_path}")
    print(f"[dataset_figures] Figure 2: {figure_2_path}")

    return {
        "figure_1": figure_1_path,
        "figure_2": figure_2_path,
    }


if __name__ == "__main__":
    from src.config_loader import load_pipeline_config

    pipeline_config = load_pipeline_config()
    generate_dataset_figures(pipeline_config)
