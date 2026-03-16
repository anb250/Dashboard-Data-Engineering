from __future__ import annotations

import argparse
import re
import sys
import unicodedata
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


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("₂", "2")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].replace({"—": np.nan, "": np.nan, "nan": np.nan})
    return out


def minmax_penalty(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min()
    hi = s.max()

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)

    return (s - lo) / (hi - lo)


def find_header_row(raw: pd.DataFrame, search_rows: int = 10) -> int:
    expected = {"brand", "model", "vehicle class"}

    best_row = None
    best_score = -1

    for i in range(min(search_rows, len(raw))):
        row_values = {normalize_text(v) for v in raw.iloc[i].tolist()}
        score = len(expected.intersection(row_values))

        if score > best_score:
            best_score = score
            best_row = i

        if expected.issubset(row_values):
            return i

    if best_row is None or best_score <= 0:
        raise ValueError("Could not detect the English header row automatically.")

    return best_row


def build_column_map(columns) -> Dict[str, str]:
    normalized = {normalize_text(col): col for col in columns}

    aliases = {
        "brand": ["brand"],
        "model": ["model"],
        "vehicle_class": ["vehicle class"],
        "co2": ["co2 g/km (max)", "co2 g/km max", "co2", "co2 g/km"],
        "engine": ["engine size l", "engine size", "engine"],
        "price": ["price proxy cad", "price proxy", "price"],
        "sales": ["ca sales 2026 jan-feb", "sales"],
    }

    out = {}

    for key, options in aliases.items():
        found = None
        for option in options:
            if option in normalized:
                found = normalized[option]
                break
        if found is None:
            raise ValueError(
                f"Could not find required column for '{key}'. "
                f"Available columns: {list(columns)}"
            )
        out[key] = found

    return out


def load_master_sheet(file_path: str | Path, sheet_name: str = "SUV主表") -> Tuple[pd.DataFrame, Dict[str, str]]:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {file_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    header_row_idx = find_header_row(raw)
    headers = [str(v).strip() if not pd.isna(v) else "" for v in raw.iloc[header_row_idx].tolist()]

    df = raw.iloc[header_row_idx + 1:].copy().reset_index(drop=True)
    df.columns = headers
    df = df.loc[:, [str(c).strip() != "" for c in df.columns]]
    df = clean_missing_markers(df)

    col_map = build_column_map(df.columns)

    numeric_candidates = [
        col_map["co2"],
        col_map["engine"],
        col_map["price"],
        col_map["sales"],
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
        "Class Avg CO2",
        "Class Avg MPG",
        "Class Avg Engine L",
        "Class Avg Cylinders",
        "Class Avg CO₂",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[col_map["brand"]] = df[col_map["brand"]].astype(str).str.strip().str.upper()
    df[col_map["model"]] = df[col_map["model"]].astype(str).str.strip()
    df[col_map["vehicle_class"]] = df[col_map["vehicle_class"]].astype(str).str.strip()

    df = df.dropna(
        subset=[
            col_map["brand"],
            col_map["model"],
            col_map["co2"],
            col_map["engine"],
            col_map["price"],
        ]
    ).reset_index(drop=True)

    return df, col_map


def filter_japanese_brands(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    out = df[df[col_map["brand"]].isin(JAPANESE_BRANDS)].copy()

    if out.empty:
        raise ValueError("No Japanese brands found after filtering.")

    return out.reset_index(drop=True)


def compute_business_scores(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    weights: Dict[str, float] | None = None
) -> pd.DataFrame:
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    total = w["co2"] + w["price"] + w["engine"]
    if total <= 0:
        raise ValueError("At least one business score weight must be greater than zero.")

    if round(total, 10) != 1.0:
        w = {k: v / total for k, v in w.items()}

    out = df.copy()

    out["co2_penalty"] = minmax_penalty(out[col_map["co2"]])
    out["price_penalty"] = minmax_penalty(out[col_map["price"]])
    out["engine_penalty"] = minmax_penalty(out[col_map["engine"]])

    out["business_score"] = (
        w["co2"] * out["co2_penalty"]
        + w["price"] * out["price_penalty"]
        + w["engine"] * out["engine_penalty"]
    )

    out["business_index"] = (1 - out["business_score"]) * 100
    out["business_rank"] = out["business_score"].rank(method="dense", ascending=True).astype(int)

    return out.sort_values(
        ["business_score", col_map["sales"]],
        ascending=[True, False]
    ).reset_index(drop=True)


def run_clustering(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    feature_cols = [col_map["price"], col_map["co2"], col_map["engine"]]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x = imputer.fit_transform(out[feature_cols])
    x = scaler.fit_transform(x)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    out["cluster"] = model.fit_predict(x)

    cluster_profile = (
        out.groupby("cluster", dropna=False)
        .agg(
            vehicles=(col_map["model"], "count"),
            avg_business_index=("business_index", "mean"),
            avg_price=(col_map["price"], "mean"),
            avg_co2=(col_map["co2"], "mean"),
            avg_engine=(col_map["engine"], "mean"),
            avg_sales=(col_map["sales"], "mean"),
        )
        .reset_index()
        .sort_values("avg_business_index", ascending=False)
        .reset_index(drop=True)
    )

    return out, cluster_profile


def build_brand_summary(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    brand = (
        df.groupby(col_map["brand"], dropna=False)
        .agg(
            models=(col_map["model"], "count"),
            avg_business_index=("business_index", "mean"),
            avg_co2=(col_map["co2"], "mean"),
            avg_price=(col_map["price"], "mean"),
            avg_engine=(col_map["engine"], "mean"),
            total_sales=(col_map["sales"], "sum"),
        )
        .reset_index()
        .rename(columns={col_map["brand"]: "Brand"})
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
            "High CO2": row["co2_gap"],
            "High price": row["price_gap"],
            "Large engine size": row["engine_gap"],
        }
        return max(gaps, key=gaps.get)

    brand["likely_underperformance_driver"] = brand.apply(cause, axis=1)
    return brand


def executive_kpis(scored_df: pd.DataFrame, brand_summary: pd.DataFrame, cluster_profile: pd.DataFrame, col_map: Dict[str, str]) -> Dict[str, object]:
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
        "best_japanese_car": f"{best_vehicle[col_map['brand']]} {best_vehicle[col_map['model']]}",
        "best_business_index": round(float(best_vehicle["business_index"]), 2),
        "lowest_co2_brand": str(lowest_co2_brand["Brand"]),
        "lowest_co2_brand_value": round(float(lowest_co2_brand["avg_co2"]), 2),
        "most_price_competitive_brand": str(most_price_competitive["Brand"]),
        "most_price_competitive_value": round(float(most_price_competitive["avg_price"]), 2),
        "worst_underperforming_brand": str(worst_underperformer["Brand"]),
        "worst_underperforming_value": round(float(worst_underperformer["avg_business_index"]), 2),
        "cluster_count": int(cluster_profile["cluster"].nunique()),
        "vehicle_count": int(scored_df.shape[0]),
        "brand_count": int(scored_df[col_map["brand"]].nunique()),
    }


def build_top10_share(scored_df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    top10 = scored_df.nsmallest(10, "business_score").copy()

    share = (
        top10.groupby(col_map["brand"])
        .size()
        .reset_index(name="top10_models")
        .rename(columns={col_map["brand"]: "Brand"})
        .sort_values(["top10_models", "Brand"], ascending=[False, True])
        .reset_index(drop=True)
    )

    share["share_of_top10"] = share["top10_models"] / 10
    return share


def save_plots(
    scored_df: pd.DataFrame,
    brand_summary: pd.DataFrame,
    col_map: Dict[str, str],
    output_dir: str | Path
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")

    top10 = scored_df.nsmallest(10, "business_score").copy()
    labels = top10[col_map["brand"]] + " " + top10[col_map["model"]]

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
    plt.scatter(scored_df[col_map["engine"]], scored_df[col_map["co2"]])
    plt.title("Engine Size vs CO2")
    plt.xlabel("Engine Size (L)")
    plt.ylabel("CO2 (g/km)")
    plt.tight_layout()
    plt.savefig(out / "engine_vs_co2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(brand_summary["Brand"], brand_summary["avg_co2"])
    plt.title("Average CO2 by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Average CO2 (g/km)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out / "avg_co2_by_brand.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        scored_df[col_map["co2"]],
        scored_df[col_map["price"]],
        c=scored_df["cluster"]
    )
    plt.title("Commercial Positioning: Price vs CO2")
    plt.xlabel("CO2 (g/km)")
    plt.ylabel("Price Proxy (CAD)")
    plt.tight_layout()
    plt.savefig(out / "price_vs_co2_clusters.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        scored_df[col_map["engine"]],
        scored_df[col_map["price"]],
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
    base, col_map = load_master_sheet(file_path)
    jp = filter_japanese_brands(base, col_map)
    scored = compute_business_scores(jp, col_map, weights=weights)
    clustered, cluster_profile = run_clustering(scored, col_map, n_clusters=n_clusters)
    brand_summary = build_brand_summary(clustered, col_map)
    kpis = executive_kpis(clustered, brand_summary, cluster_profile, col_map)
    top10_share = build_top10_share(clustered, col_map)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clustered.to_csv(out / "japanese_suv_business_model_output.csv", index=False)
    brand_summary.to_csv(out / "brand_summary.csv", index=False)
    cluster_profile.to_csv(out / "cluster_profile.csv", index=False)
    top10_share.to_csv(out / "top10_brand_share.csv", index=False)

    pd.DataFrame([kpis]).to_csv(out / "executive_kpis.csv", index=False)

    save_plots(clustered, brand_summary, col_map, out)

    return {
        "scored_df": clustered,
        "brand_summary": brand_summary,
        "cluster_profile": cluster_profile,
        "top10_share": top10_share,
        "kpis": kpis,
        "output_dir": str(out),
        "col_map": col_map,
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
