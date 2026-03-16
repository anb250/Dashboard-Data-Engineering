from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


JAPANESE_BRANDS = {
    "ACURA", "HONDA", "INFINITI", "LEXUS", "MAZDA",
    "MITSUBISHI", "NISSAN", "SUBARU", "SUZUKI", "TOYOTA"
}

DEFAULT_WEIGHTS = {
    "co2": 0.45,
    "price": 0.35,
    "engine": 0.20,
}

REQUIRED_COLUMNS = {
    "brand": "Brand",
    "model": "Model",
    "vehicle_class": "Vehicle Class",
    "co2": "CO₂ g/km (max)",
    "engine": "Engine Size L",
    "price": "Price Proxy CAD",
    "sales": "CA Sales 2026 Jan–Feb",
}

OPTIONAL_COLUMNS = [
    "CA Brand Total",
    "CA Sales Share",
    "Fuel Type [CA]",
    "Transmission [CA]",
    "Class Avg CO₂",
    "Class Avg MPG",
    "Class Avg Engine L",
    "Class Avg Cylinders",
    "Demand Target Units",
    "Revenue Target CAD",
    "Brand Benchmark CAD",
    "Price Index (class)",
    "Powertrain Type",
    "Cylinders",
    "Max Power PS (max)",
    "Max Torque Nm (max)",
    "Drive Config",
    "Seats (max)",
]


def minmax_penalty(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min()
    hi = s.max()

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)

    return (s - lo) / (hi - lo)


def _normalize_header_value(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _find_english_header_row(raw: pd.DataFrame, search_rows: int = 8) -> int:
    """
    Find the row that contains the actual English field headers.
    """
    expected = {"Brand", "Model", "Vehicle Class"}

    best_row = None
    best_score = -1

    for i in range(min(search_rows, len(raw))):
        row_values = {_normalize_header_value(v) for v in raw.iloc[i].tolist()}
        score = len(expected.intersection(row_values))

        if score > best_score:
            best_score = score
            best_row = i

        if expected.issubset(row_values):
            return i

    if best_row is None or best_score <= 0:
        raise ValueError(
            "Could not detect the English header row automatically. "
            "Please inspect the first 8 rows of the sheet."
        )

    return best_row


def load_master_sheet(file_path: str | Path, sheet_name: str = "SUV主表") -> pd.DataFrame:
    """
    Load the bilingual SUV master sheet and automatically detect the English header row.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {file_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    header_row_idx = _find_english_header_row(raw)
    english_header = [_normalize_header_value(v) for v in raw.iloc[header_row_idx].tolist()]

    df = raw.iloc[header_row_idx + 1:].copy().reset_index(drop=True)
    df.columns = english_header

    df = df.loc[:, [str(c).strip() != "" for c in df.columns]]

    for col in df.columns:
        if isinstance(col, str):
            df[col] = df[col].replace({"—": np.nan, "": np.nan, "nan": np.nan})

    missing = [excel_name for excel_name in REQUIRED_COLUMNS.values() if excel_name not in df.columns]
    if missing:
        preview_rows = raw.head(6).fillna("").astype(str)
        raise ValueError(
            f"Missing required columns in workbook: {missing}\n"
            f"Detected header row index: {header_row_idx}\n"
            f"Available columns: {list(df.columns)}\n\n"
            f"Top rows preview:\n{preview_rows.to_string(index=True, header=False)}"
        )

    keep = list(REQUIRED_COLUMNS.values()) + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep].copy()

    numeric_cols = [
        REQUIRED_COLUMNS["co2"],
        REQUIRED_COLUMNS["engine"],
        REQUIRED_COLUMNS["price"],
        REQUIRED_COLUMNS["sales"],
        "CA Brand Total",
        "CA Sales Share",
        "Demand Target Units",
        "Revenue Target CAD",
        "Brand Benchmark CAD",
        "Price Index (class)",
        "Cylinders",
        "Max Power PS (max)",
        "Max Torque Nm (max)",
        "Seats (max)",
        "Class Avg CO₂",
        "Class Avg MPG",
        "Class Avg Engine L",
        "Class Avg Cylinders",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Brand"] = df["Brand"].astype(str).str.strip().str.upper()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Vehicle Class"] = df["Vehicle Class"].astype(str).str.strip()

    df = df.dropna(
        subset=[
            "Brand",
            "Model",
            REQUIRED_COLUMNS["co2"],
            REQUIRED_COLUMNS["engine"],
            REQUIRED_COLUMNS["price"],
        ]
    ).reset_index(drop=True)

    return df


def filter_japanese_brands(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["Brand"].isin(JAPANESE_BRANDS)].copy()

    if out.empty:
        raise ValueError("No Japanese brands found after filtering.")

    return out.reset_index(drop=True)


def compute_business_scores(df: pd.DataFrame, weights: Dict[str, float] | None = None) -> pd.DataFrame:
    """
    Lower CO2, lower price, and lower engine size are treated as better.
    A weighted penalty score is converted into a 0 to 100 business index.
    """
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    total = w["co2"] + w["price"] + w["engine"]
    if total <= 0:
        raise ValueError("At least one business score weight must be greater than zero.")

    if round(total, 10) != 1.0:
        w = {k: v / total for k, v in w.items()}

    out = df.copy()

    out["co2_penalty"] = minmax_penalty(out[REQUIRED_COLUMNS["co2"]])
    out["price_penalty"] = minmax_penalty(out[REQUIRED_COLUMNS["price"]])
    out["engine_penalty"] = minmax_penalty(out[REQUIRED_COLUMNS["engine"]])

    out["business_score"] = (
        w["co2"] * out["co2_penalty"]
        + w["price"] * out["price_penalty"]
        + w["engine"] * out["engine_penalty"]
    )

    out["business_index"] = (1 - out["business_score"]) * 100
    out["business_rank"] = out["business_score"].rank(method="dense", ascending=True).astype(int)

    return out.sort_values(
        ["business_score", REQUIRED_COLUMNS["sales"]],
        ascending=[True, False]
    ).reset_index(drop=True)


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    feature_cols = [
        REQUIRED_COLUMNS["price"],
        REQUIRED_COLUMNS["co2"],
        REQUIRED_COLUMNS["engine"],
    ]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x = imputer.fit_transform(out[feature_cols])
    x = scaler.fit_transform(x)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    out["cluster"] = model.fit_predict(x)

    cluster_profile = (
        out.groupby("cluster", dropna=False)
        .agg(
            vehicles=("Model", "count"),
            avg_business_index=("business_index", "mean"),
            avg_price=(REQUIRED_COLUMNS["price"], "mean"),
            avg_co2=(REQUIRED_COLUMNS["co2"], "mean"),
            avg_engine=(REQUIRED_COLUMNS["engine"], "mean"),
            avg_sales=(REQUIRED_COLUMNS["sales"], "mean"),
        )
        .reset_index()
        .sort_values("avg_business_index", ascending=False)
        .reset_index(drop=True)
    )

    return out, cluster_profile


def build_brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    brand = (
        df.groupby("Brand", dropna=False)
        .agg(
            models=("Model", "count"),
            avg_business_index=("business_index", "mean"),
            avg_co2=(REQUIRED_COLUMNS["co2"], "mean"),
            avg_price=(REQUIRED_COLUMNS["price"], "mean"),
            avg_engine=(REQUIRED_COLUMNS["engine"], "mean"),
            total_sales=(REQUIRED_COLUMNS["sales"], "sum"),
        )
        .reset_index()
        .sort_values(["avg_business_index", "total_sales"], ascending=[False, False])
        .reset_index(drop=True)
    )

    brand["underperforming_flag"] = np.where(
        brand["avg_business_index"] < brand["avg_business_index"].median(),
        "Underperforming",
        "Stronger performing",
    )

    overall_co2 = brand["avg_co2"].mean()
    overall_price = brand["avg_price"].mean()
    overall_engine = brand["avg_engine"].mean()

    brand["co2_gap"] = brand["avg_co2"] - overall_co2
    brand["price_gap"] = brand["avg_price"] - overall_price
    brand["engine_gap"] = brand["avg_engine"] - overall_engine

    def cause(row: pd.Series) -> str:
        gaps = {
            "High CO₂": row["co2_gap"],
            "High price": row["price_gap"],
            "Large engine size": row["engine_gap"],
        }
        return max(gaps, key=gaps.get)

    brand["likely_underperformance_driver"] = brand.apply(cause, axis=1)
    return brand


def executive_kpis(
    scored_df: pd.DataFrame,
    brand_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame
) -> Dict[str, object]:
    best_vehicle = scored_df.iloc[0]

    lowest_co2_brand = brand_summary.sort_values(
        ["avg_co2", "avg_business_index"],
        ascending=[True, False]
    ).iloc[0]

    most_price_competitive = brand_summary.sort_values(
        ["avg_price", "avg_business_index"],
        ascending=[True, False]
    ).iloc[0]

    worst_underperformer = brand_summary.sort_values(
        ["avg_business_index", "avg_co2"],
        ascending=[True, False]
    ).iloc[0]

    return {
        "best_japanese_car": f"{best_vehicle['Brand']} {best_vehicle['Model']}",
        "best_business_index": round(float(best_vehicle["business_index"]), 2),
        "lowest_co2_brand": str(lowest_co2_brand["Brand"]),
        "lowest_co2_brand_value": round(float(lowest_co2_brand["avg_co2"]), 2),
        "most_price_competitive_brand": str(most_price_competitive["Brand"]),
        "most_price_competitive_value": round(float(most_price_competitive["avg_price"]), 2),
        "worst_underperforming_brand": str(worst_underperformer["Brand"]),
        "worst_underperforming_value": round(float(worst_underperformer["avg_business_index"]), 2),
        "cluster_count": int(cluster_profile["cluster"].nunique()),
        "vehicle_count": int(scored_df.shape[0]),
        "brand_count": int(scored_df["Brand"].nunique()),
    }


def build_top10_share(scored_df: pd.DataFrame) -> pd.DataFrame:
    top10 = scored_df.nsmallest(10, "business_score").copy()

    share = (
        top10.groupby("Brand")
        .size()
        .reset_index(name="top10_models")
        .sort_values(["top10_models", "Brand"], ascending=[False, True])
        .reset_index(drop=True)
    )

    share["share_of_top10"] = share["top10_models"] / 10
    return share


def save_plots(
    scored_df: pd.DataFrame,
    brand_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    output_dir: str | Path
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")

    top10 = scored_df.nsmallest(10, "business_score").copy()
    labels = top10["Brand"] + " " + top10["Model"]

    plt.figure(figsize=(12, 6))
    plt.barh(labels, top10["business_index"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Japanese SUVs by Business Index")
    plt.xlabel("Business Index")
    plt.ylabel("Vehicle")
    plt.tight_layout()
    plt.savefig(out / "top10_business_index.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(brand_summary["Brand"], brand_summary["avg_business_index"])
    plt.title("Average Business Index by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Average Business Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out / "brand_business_index.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scored_df[REQUIRED_COLUMNS["engine"]], scored_df[REQUIRED_COLUMNS["co2"]])
    plt.title("Engine Size vs CO₂")
    plt.xlabel("Engine Size (L)")
    plt.ylabel("CO₂ (g/km)")
    plt.tight_layout()
    plt.savefig(out / "engine_vs_co2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(brand_summary["Brand"], brand_summary["avg_co2"])
    plt.title("Average CO₂ by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Average CO₂ (g/km)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out / "avg_co2_by_brand.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        scored_df[REQUIRED_COLUMNS["co2"]],
        scored_df[REQUIRED_COLUMNS["price"]],
        c=scored_df["cluster"]
    )
    plt.title("Commercial Positioning: Price vs CO₂")
    plt.xlabel("CO₂ (g/km)")
    plt.ylabel("Price Proxy (CAD)")
    plt.tight_layout()
    plt.savefig(out / "price_vs_co2_clusters.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        scored_df[REQUIRED_COLUMNS["engine"]],
        scored_df[REQUIRED_COLUMNS["price"]],
        c=scored_df["cluster"]
    )
    plt.title("Cluster-Based Positioning Map")
    plt.xlabel("Engine Size (L)")
    plt.ylabel("Price Proxy (CAD)")
    plt.tight_layout()
    plt.savefig(out / "cluster_positioning_map.png", dpi=300)
    plt.close()

    under = brand_summary.sort_values("avg_business_index", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(under["Brand"], under["avg_business_index"])
    plt.title("Underperforming Brand Comparison")
    plt.xlabel("Average Business Index")
    plt.ylabel("Brand")
    plt.tight_layout()
    plt.savefig(out / "underperforming_brands.png", dpi=300)
    plt.close()


def run_business_model(
    file_path: str | Path,
    output_dir: str | Path = "business_outputs",
    n_clusters: int = 3,
    weights: Dict[str, float] | None = None,
) -> Dict[str, object]:
    base = load_master_sheet(file_path)
    jp = filter_japanese_brands(base)
    scored = compute_business_scores(jp, weights=weights)
    clustered, cluster_profile = run_clustering(scored, n_clusters=n_clusters)
    brand_summary = build_brand_summary(clustered)
    kpis = executive_kpis(clustered, brand_summary, cluster_profile)
    top10_share = build_top10_share(clustered)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clustered.to_csv(out / "japanese_suv_business_model_output.csv", index=False)
    brand_summary.to_csv(out / "brand_summary.csv", index=False)
    cluster_profile.to_csv(out / "cluster_profile.csv", index=False)
    top10_share.to_csv(out / "top10_brand_share.csv", index=False)
    pd.DataFrame([kpis]).to_csv(out / "executive_kpis.csv", index=False)

    save_plots(clustered, brand_summary, cluster_profile, out)

    return {
        "scored_df": clustered,
        "brand_summary": brand_summary,
        "cluster_profile": cluster_profile,
        "top10_share": top10_share,
        "kpis": kpis,
        "output_dir": str(out),
    }


def main(file_path=None, output_dir="business_outputs", clusters=3):
    if file_path is None:
        parser = argparse.ArgumentParser(description="Run Japanese SUV business model analysis.")
        parser.add_argument("--file", required=True, help="Path to the Excel workbook")
        parser.add_argument("--output", default="business_outputs", help="Output folder for csv and png files")
        parser.add_argument("--clusters", type=int, default=3, help="Number of KMeans clusters")
        args = parser.parse_args()

        file_path = args.file
        output_dir = args.output
        clusters = args.clusters

    results = run_business_model(file_path, output_dir=output_dir, n_clusters=clusters)
    print("Business model completed.")
    print(pd.DataFrame([results["kpis"]]).to_string(index=False))
    return results


def _running_in_notebook() -> bool:
    return "ipykernel" in sys.modules


if __name__ == "__main__":
    if _running_in_notebook():
        print("Notebook environment detected. No automatic CLI run was performed.")
        print("Run this in a new cell:")
        print()
        print('results = main(file_path="Canada_JP_SUV_FILLED_FINAL_v2.xlsx", output_dir="business_outputs", clusters=3)')
        print()
    else:
        main()



Notebook environment detected. No automatic CLI run was performed.
Run this in a new cell:

results = main(file_path="Canada_JP_SUV_FILLED_FINAL_v2.xlsx", output_dir="business_outputs", clusters=3)
