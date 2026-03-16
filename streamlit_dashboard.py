from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from business_model import run_business_model, DEFAULT_WEIGHTS


st.set_page_config(
    page_title="Japanese SUV Portfolio Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Japanese SUV Portfolio Strategy Dashboard")
st.caption(
    "Business-facing portfolio analysis for executive stakeholders, with full analytical detail available for academic review."
)


with st.sidebar:
    st.header("Analysis Controls")

    uploaded = st.file_uploader("Upload the Excel workbook", type=["xlsx"])

    clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=6,
        value=3,
        step=1,
    )

    st.markdown("### Business Score Weights")
    co2_w = st.slider("CO2 weight", 0.0, 1.0, float(DEFAULT_WEIGHTS["co2"]), 0.05)
    price_w = st.slider("Price weight", 0.0, 1.0, float(DEFAULT_WEIGHTS["price"]), 0.05)
    engine_w = st.slider("Engine size weight", 0.0, 1.0, float(DEFAULT_WEIGHTS["engine"]), 0.05)

    weight_total = co2_w + price_w + engine_w
    if weight_total == 0:
        st.error("At least one weight must be greater than zero.")
        st.stop()

    weights = {
        "co2": co2_w / weight_total,
        "price": price_w / weight_total,
        "engine": engine_w / weight_total,
    }


@st.cache_data(show_spinner=False)
def load_results(file_bytes: bytes, filename: str, n_clusters: int, weights: dict):
    temp_dir = Path("streamlit_temp")
    temp_dir.mkdir(exist_ok=True)

    workbook_path = temp_dir / filename
    workbook_path.write_bytes(file_bytes)

    output_dir = temp_dir / "business_outputs"

    return run_business_model(
        file_path=workbook_path,
        output_dir=output_dir,
        n_clusters=n_clusters,
        weights=weights,
    )


def coerce_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def prepare_scatter_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str | None = None,
    color_col: str | None = None,
    symbol_col: str | None = None,
    hover_cols: list[str] | None = None,
) -> pd.DataFrame:
    plot_df = df.copy()

    numeric_cols = [x_col, y_col]
    if size_col is not None:
        numeric_cols.append(size_col)
    if "business_index" in plot_df.columns and "business_index" not in numeric_cols:
        numeric_cols.append("business_index")

    plot_df = coerce_numeric_columns(plot_df, numeric_cols)

    required = [x_col, y_col]
    if size_col is not None:
        required.append(size_col)

    plot_df = plot_df.dropna(subset=required).copy()

    if size_col is not None and size_col in plot_df.columns:
        plot_df = plot_df[plot_df[size_col] > 0].copy()

    if color_col is not None and color_col in plot_df.columns:
        plot_df[color_col] = plot_df[color_col].astype(str)

    if symbol_col is not None and symbol_col in plot_df.columns:
        plot_df[symbol_col] = plot_df[symbol_col].astype(str)

    if "cluster" in plot_df.columns:
        plot_df["cluster"] = plot_df["cluster"].astype(str)

    if hover_cols:
        protected_cols = {x_col, y_col}
        if size_col is not None:
            protected_cols.add(size_col)

        for col in hover_cols:
            if col in plot_df.columns and col not in protected_cols:
                plot_df[col] = plot_df[col].astype(str)

    return plot_df.reset_index(drop=True)


def safe_plotly_chart(fig, fallback_df: pd.DataFrame | None = None, message: str | None = None):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(message or f"Chart could not be rendered: {e}")
        if fallback_df is not None and not fallback_df.empty:
            st.dataframe(fallback_df.head(25), use_container_width=True)


if uploaded is None:
    st.info("Upload the workbook to generate the dashboard.")
    st.stop()

try:
    results = load_results(uploaded.getvalue(), uploaded.name, clusters, weights)
except Exception as e:
    st.error("The dashboard could not process the workbook.")
    st.exception(e)
    st.stop()

scored_df = results["scored_df"]
brand_summary = results["brand_summary"]
cluster_profile = results["cluster_profile"]
top10_share = results["top10_share"]
kpis = results["kpis"]
col_map = results["col_map"]

brand_col = col_map["brand"]
model_col = col_map["model"]
co2_col = col_map["co2"]
engine_col = col_map["engine"]
price_col = col_map["price"]
sales_col = col_map["sales"]

scored_df = coerce_numeric_columns(
    scored_df,
    [co2_col, engine_col, price_col, sales_col, "business_index", "business_score"]
)
brand_summary = coerce_numeric_columns(
    brand_summary,
    ["avg_business_index", "avg_price", "avg_co2", "avg_engine", "total_sales", "co2_gap", "price_gap", "engine_gap"]
)
cluster_profile = coerce_numeric_columns(
    cluster_profile,
    ["avg_business_index", "avg_price", "avg_co2", "avg_engine", "avg_sales", "vehicles"]
)

if "cluster" in scored_df.columns:
    scored_df["cluster"] = scored_df["cluster"].astype(str)


k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Best Japanese car",
    kpis["best_japanese_car"],
    f"Index {kpis['best_business_index']}"
)

k2.metric(
    "Lowest CO2 brand",
    kpis["lowest_co2_brand"],
    f"{kpis['lowest_co2_brand_value']} g/km"
)

k3.metric(
    "Most price-competitive brand",
    kpis["most_price_competitive_brand"],
    f"CAD {kpis['most_price_competitive_value']:,.0f}"
)

k4.metric(
    "Worst underperforming brand",
    kpis["worst_underperforming_brand"],
    f"Index {kpis['worst_underperforming_value']}"
)

k5.metric(
    "Number of clusters",
    kpis["cluster_count"],
    f"{kpis['vehicle_count']} vehicles"
)


exec_tab, portfolio_tab, efficiency_tab, commercial_tab, diagnosis_tab, professor_tab = st.tabs([
    "Executive View",
    "Portfolio Performance",
    "Efficiency and Emissions",
    "Commercial Competitiveness",
    "Underperformance Diagnosis",
    "Professor Data Room",
])


with exec_tab:
    st.subheader("Executive Summary")

    c1, c2 = st.columns([1.3, 1])

    with c1:
        top10 = scored_df.nsmallest(10, "business_score").copy()
        top10["Vehicle"] = top10[brand_col].astype(str) + " " + top10[model_col].astype(str)

        fig = px.bar(
            top10.sort_values("business_index", ascending=True),
            x="business_index",
            y="Vehicle",
            orientation="h",
            title="Top 10 Japanese SUVs by Business Index",
            labels={
                "business_index": "Business Index",
                "Vehicle": "Vehicle",
            },
            color=brand_col,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        safe_plotly_chart(fig, top10, "Top 10 chart could not be rendered.")

    with c2:
        if not top10_share.empty:
            fig = px.pie(
                top10_share,
                names="Brand",
                values="top10_models",
                title="Brand Share of Top 10 Ranked Vehicles",
            )
            safe_plotly_chart(fig, top10_share, "Top 10 brand share could not be rendered.")

    st.markdown(
        f"""
        **Management interpretation**

        The current top-ranked vehicle is **{kpis['best_japanese_car']}** under the selected weighting scheme.
        **{kpis['lowest_co2_brand']}** leads on average emissions performance.
        **{kpis['most_price_competitive_brand']}** is the strongest brand on average price positioning.
        **{kpis['worst_underperforming_brand']}** shows the weakest average business index and should be reviewed most closely.
        """
    )


with portfolio_tab:
    st.subheader("Portfolio Performance")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            brand_summary,
            x="Brand",
            y="avg_business_index",
            color="underperforming_flag",
            title="Average Business Index by Brand",
            labels={"avg_business_index": "Average Business Index"},
        )
        safe_plotly_chart(fig, brand_summary, "Brand performance chart could not be rendered.")

    with c2:
        fig = px.bar(
            brand_summary,
            x="Brand",
            y="total_sales",
            color="Brand",
            title="Total Sales by Brand",
            labels={"total_sales": "CA Sales 2026 Jan-Feb"},
        )
        safe_plotly_chart(fig, brand_summary, "Brand sales chart could not be rendered.")

    st.markdown("### Portfolio Summary Table")
    st.dataframe(
        brand_summary[
            [
                "Brand",
                "models",
                "avg_business_index",
                "avg_price",
                "avg_co2",
                "avg_engine",
                "total_sales",
                "underperforming_flag",
            ]
        ],
        use_container_width=True,
    )


with efficiency_tab:
    st.subheader("Efficiency and Emissions")

    c1, c2 = st.columns(2)

    with c1:
        plot_df_3 = prepare_scatter_df(
            scored_df,
            x_col=engine_col,
            y_col=co2_col,
            size_col="business_index",
            color_col=brand_col,
            hover_cols=[model_col, price_col, sales_col],
        )

        fig = px.scatter(
            plot_df_3,
            x=engine_col,
            y=co2_col,
            color=brand_col,
            size="business_index",
            hover_data=[model_col, price_col, sales_col, "business_index"],
            title="Engine Size versus CO2",
            labels={
                engine_col: "Engine Size (L)",
                co2_col: "CO2 (g/km)",
            },
        )
        safe_plotly_chart(fig, plot_df_3, "Efficiency scatter could not be rendered.")

    with c2:
        co2_sorted = brand_summary.sort_values("avg_co2", ascending=True)
        fig = px.bar(
            co2_sorted,
            x="Brand",
            y="avg_co2",
            color="Brand",
            title="Average CO2 by Brand",
            labels={"avg_co2": "Average CO2 (g/km)"},
        )
        safe_plotly_chart(fig, co2_sorted, "Average CO2 chart could not be rendered.")

    st.markdown(
        "This section shows how engine size and emissions profile vary across the Japanese SUV portfolio."
    )


with commercial_tab:
    st.subheader("Commercial Competitiveness")

    c1, c2 = st.columns(2)

    with c1:
        plot_df_1 = prepare_scatter_df(
            scored_df,
            x_col=co2_col,
            y_col=price_col,
            size_col=sales_col,
            color_col="cluster",
            symbol_col=brand_col,
            hover_cols=[brand_col, model_col, engine_col, "business_index"],
        )

        fig = px.scatter(
            plot_df_1,
            x=co2_col,
            y=price_col,
            color="cluster",
            symbol=brand_col,
            size=sales_col,
            hover_data=[brand_col, model_col, engine_col, "business_index"],
            title="Price versus CO2",
            labels={
                co2_col: "CO2 (g/km)",
                price_col: "Price Proxy (CAD)",
            },
        )
        safe_plotly_chart(fig, plot_df_1, "Price versus CO2 chart could not be rendered.")

    with c2:
        plot_df_2 = prepare_scatter_df(
            scored_df,
            x_col=engine_col,
            y_col=price_col,
            size_col="business_index",
            color_col="cluster",
            symbol_col=brand_col,
            hover_cols=[model_col, co2_col, sales_col],
        )

        fig = px.scatter(
            plot_df_2,
            x=engine_col,
            y=price_col,
            color="cluster",
            symbol=brand_col,
            size="business_index",
            hover_data=[model_col, co2_col, sales_col],
            title="Cluster-Based Positioning Map",
            labels={
                engine_col: "Engine Size (L)",
                price_col: "Price Proxy (CAD)",
            },
        )
        safe_plotly_chart(fig, plot_df_2, "Cluster positioning chart could not be rendered.")

    st.markdown("### Cluster Profile")
    st.dataframe(cluster_profile, use_container_width=True)


with diagnosis_tab:
    st.subheader("Underperformance Diagnosis")

    c1, c2 = st.columns(2)

    with c1:
        under = brand_summary.sort_values("avg_business_index", ascending=True)
        fig = px.bar(
            under,
            x="Brand",
            y="avg_business_index",
            color="likely_underperformance_driver",
            title="Likely Cause by Brand",
            labels={"avg_business_index": "Average Business Index"},
        )
        safe_plotly_chart(fig, under, "Underperformance driver chart could not be rendered.")

    with c2:
        driver_counts = brand_summary["likely_underperformance_driver"].value_counts().reset_index()
        driver_counts.columns = ["Driver", "Brands"]

        fig = px.pie(
            driver_counts,
            names="Driver",
            values="Brands",
            title="Distribution of Underperformance Drivers",
        )
        safe_plotly_chart(fig, driver_counts, "Driver distribution chart could not be rendered.")

    st.markdown("### Brand Diagnosis Table")
    st.dataframe(
        brand_summary[
            [
                "Brand",
                "avg_business_index",
                "avg_co2",
                "avg_price",
                "avg_engine",
                "co2_gap",
                "price_gap",
                "engine_gap",
                "likely_underperformance_driver",
                "underperforming_flag",
            ]
        ],
        use_container_width=True,
    )


with professor_tab:
    st.subheader("Professor Data Room")
    st.markdown(
        "This section preserves the full analytical detail behind the executive dashboard."
    )

    with st.expander("Vehicle-level scored output", expanded=False):
        st.dataframe(scored_df, use_container_width=True)

    with st.expander("Brand summary", expanded=False):
        st.dataframe(brand_summary, use_container_width=True)

    with st.expander("Cluster profile", expanded=False):
        st.dataframe(cluster_profile, use_container_width=True)

    with st.expander("Top 10 brand share", expanded=False):
        st.dataframe(top10_share, use_container_width=True)

    with st.expander("Method note", expanded=False):
        st.markdown(
            """
            **Business model logic**

            - Lower CO2 is treated as better.
            - Lower price is treated as more commercially competitive.
            - Lower engine size is treated as more efficient.
            - The business score is a weighted penalty score converted into a 0 to 100 business index.
            - KMeans clustering groups vehicles into commercially comparable segments.
            - Underperformance is diagnosed from the largest adverse gap in CO2, price, or engine size relative to the multi-brand average.
            """
        )

    with st.expander("Download tables", expanded=False):
        st.download_button(
            "Download scored vehicle output CSV",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="japanese_suv_business_model_output.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download brand summary CSV",
            data=brand_summary.to_csv(index=False).encode("utf-8"),
            file_name="brand_summary.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download cluster profile CSV",
            data=cluster_profile.to_csv(index=False).encode("utf-8"),
            file_name="cluster_profile.csv",
            mime="text/csv",
        )


st.markdown("---")
st.caption(
    "Dashboard outputs update automatically when the workbook, clustering choice, or business score weights change."
)
