import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Vehicle Emissions and Scenario Dashboard",
    layout="wide"
)

# ============================================================
# THEME-AWARE STYLING
# ============================================================

THEME_BASE = st.get_option("theme.base") or "light"
IS_DARK = THEME_BASE == "dark"

FONT_COLOR = "#F9FAFB" if IS_DARK else "#111827"
SUBTLE_TEXT = "#D1D5DB" if IS_DARK else "#6B7280"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(255,255,255,0.03)" if IS_DARK else "#FFFFFF"
GRID_COLOR = "rgba(255,255,255,0.12)" if IS_DARK else "rgba(0,0,0,0.08)"
AXIS_COLOR = "#E5E7EB" if IS_DARK else "#374151"
PLOT_TEMPLATE = "plotly_dark" if IS_DARK else "plotly_white"
COLORWAY = ["#60A5FA", "#34D399", "#FBBF24", "#F87171", "#A78BFA", "#22D3EE"]

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        color: {FONT_COLOR};
    }}
    h1, h2, h3, h4 {{
        color: {FONT_COLOR};
    }}
    div[data-testid="stMetricValue"] {{
        color: {FONT_COLOR};
    }}
    div[data-testid="stMetricLabel"] {{
        color: {SUBTLE_TEXT};
    }}
    div[data-testid="stCaptionContainer"] {{
        color: {SUBTLE_TEXT};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Vehicle Emissions, Demand, Revenue, and Scenario Dashboard")
st.caption("Interactive dashboard for the final coursework modelling framework")

OUTPUT_DIR = "dashboard_outputs"

# ============================================================
# LOAD SAVED OUTPUTS
# ============================================================

@st.cache_data
def load_outputs():
    return {
        "cleaned_data": pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv")),
        "emissions_results": pd.read_csv(os.path.join(OUTPUT_DIR, "emissions_results.csv")),
        "demand_results": pd.read_csv(os.path.join(OUTPUT_DIR, "demand_results.csv")),
        "revenue_results": pd.read_csv(os.path.join(OUTPUT_DIR, "revenue_results.csv")),
        "emissions_samples": pd.read_csv(os.path.join(OUTPUT_DIR, "emissions_sample_predictions.csv")),
        "demand_samples": pd.read_csv(os.path.join(OUTPUT_DIR, "demand_sample_predictions.csv")),
        "revenue_samples": pd.read_csv(os.path.join(OUTPUT_DIR, "revenue_sample_predictions.csv")),
        "emissions_importance": pd.read_csv(os.path.join(OUTPUT_DIR, "emissions_feature_importance.csv")),
        "demand_importance": pd.read_csv(os.path.join(OUTPUT_DIR, "demand_feature_importance.csv")),
        "revenue_importance": pd.read_csv(os.path.join(OUTPUT_DIR, "revenue_feature_importance.csv")),
        "scenario_results": pd.read_csv(os.path.join(OUTPUT_DIR, "scenario_results.csv")),
        "summary_metrics": pd.read_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
    }

outputs = load_outputs()
df = outputs["cleaned_data"]

# ============================================================
# VISUAL HELPERS
# ============================================================

def apply_clean_layout(fig, title):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        title_x=0.02,
        font=dict(size=14, color=FONT_COLOR),
        title_font=dict(size=22, color=FONT_COLOR),
        margin=dict(l=40, r=30, t=70, b=40),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        colorway=COLORWAY,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=FONT_COLOR)
        )
    )
    fig.update_xaxes(
        showgrid=False,
        color=AXIS_COLOR,
        linecolor=AXIS_COLOR,
        tickfont=dict(color=FONT_COLOR),
        title_font=dict(color=FONT_COLOR)
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        color=AXIS_COLOR,
        linecolor=AXIS_COLOR,
        tickfont=dict(color=FONT_COLOR),
        title_font=dict(color=FONT_COLOR),
        zerolinecolor=GRID_COLOR
    )
    return fig

def leaderboard_chart(results_df, metric, title):
    plot_df = results_df.sort_values(metric, ascending=True).copy()
    fig = px.bar(plot_df, x=metric, y="Technique", orientation="h", text=metric)
    fig.update_traces(textposition="outside")
    fig = apply_clean_layout(fig, title)
    fig.update_yaxes(categoryorder="total ascending")
    return fig

def metric_bar_figure(results_df, metric, title):
    fig = px.bar(results_df, x="Technique", y=metric, text=metric)
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-25)
    return apply_clean_layout(fig, title)

def model_metric_heatmap(results_df, title):
    heat_df = results_df.set_index("Technique")[["Test RMSE", "Test MAE", "Test R²", "CV RMSE Mean"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns,
            y=heat_df.index,
            text=np.round(heat_df.values, 3),
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True
        )
    )
    fig.update_traces(textfont={"color": "#111827"})
    return apply_clean_layout(fig, title)

def actual_vs_pred_figure(sample_df, title, y_label):
    fig = px.scatter(sample_df, x="Actual", y="Predicted", opacity=0.55)
    low = min(sample_df["Actual"].min(), sample_df["Predicted"].min())
    high = max(sample_df["Actual"].max(), sample_df["Predicted"].max())
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect fit"))
    fig = apply_clean_layout(fig, title)
    fig.update_xaxes(title=f"Actual {y_label}")
    fig.update_yaxes(title=f"Predicted {y_label}")
    return fig

def residual_figure(sample_df, title):
    fig = px.scatter(sample_df, x="Predicted", y="Residual", opacity=0.55)
    fig.add_hline(y=0)
    fig = apply_clean_layout(fig, title)
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Residual")
    return fig

def residual_distribution_figure(sample_df, title):
    fig = px.histogram(sample_df, x="Residual", nbins=40, opacity=0.85)
    fig.add_vline(x=0)
    fig = apply_clean_layout(fig, title)
    fig.update_xaxes(title="Residual")
    fig.update_yaxes(title="Count")
    return fig

def feature_importance_figure(importance_df, title):
    if importance_df.empty:
        return None
    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return apply_clean_layout(fig, title)

def sample_predictions_figure(sample_df, title):
    plot_df = sample_df.head(20).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Sample Index"], y=plot_df["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=plot_df["Sample Index"], y=plot_df["Predicted"], mode="lines+markers", name="Predicted"))
    fig = apply_clean_layout(fig, title)
    fig.update_xaxes(title="Sample index")
    fig.update_yaxes(title="Value")
    return fig

def scenario_combo_chart(scenario_df):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Predicted CO2 by vehicle class", "Demand and revenue by vehicle class"),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )

    fig.add_trace(go.Bar(x=scenario_df["Vehicle Class"], y=scenario_df["Predicted CO2 per Vehicle"], name="Predicted CO2"), row=1, col=1)
    fig.add_trace(go.Bar(x=scenario_df["Vehicle Class"], y=scenario_df["Predicted Demand"], name="Predicted Demand"), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=scenario_df["Vehicle Class"], y=scenario_df["Predicted Revenue"], mode="lines+markers", name="Predicted Revenue"), row=1, col=2, secondary_y=True)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Scenario simulator outputs by vehicle class",
        title_x=0.02,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(color=FONT_COLOR)),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR)
    )

    fig.update_yaxes(title_text="CO2 per vehicle", row=1, col=1, color=AXIS_COLOR)
    fig.update_yaxes(title_text="Demand", row=1, col=2, secondary_y=False, color=AXIS_COLOR)
    fig.update_yaxes(title_text="Revenue", row=1, col=2, secondary_y=True, color=AXIS_COLOR)
    fig.update_xaxes(color=AXIS_COLOR)
    return fig

def weighted_emissions_chart(scenario_df):
    plot_df = scenario_df.copy()
    plot_df["Weighted Emissions Contribution"] = plot_df["Production Share"] * plot_df["Predicted CO2 per Vehicle"]
    fig = px.bar(plot_df, x="Vehicle Class", y="Weighted Emissions Contribution", text="Weighted Emissions Contribution")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return apply_clean_layout(fig, "Fleet weighted emissions contribution by vehicle class")

# ============================================================
# LIGHTWEIGHT SCENARIO TAB
# ============================================================

st.sidebar.subheader("Scenario mix")
compact = st.sidebar.slider("Compact share", 0.0, 1.0, 0.30, 0.01)
mid_size = st.sidebar.slider("Mid-size share", 0.0, 1.0, 0.20, 0.01)
small_suv = st.sidebar.slider("Small SUV share", 0.0, 1.0, 0.25, 0.01)
subcompact = st.sidebar.slider("Subcompact share", 0.0, 1.0, 0.10, 0.01)
standard_suv = st.sidebar.slider("Standard SUV share", 0.0, 1.0, 0.10, 0.01)
two_seater = st.sidebar.slider("Two-seater share", 0.0, 1.0, 0.05, 0.01)

scenario_total = compact + mid_size + small_suv + subcompact + standard_suv + two_seater
st.sidebar.write(f"Scenario share total: {scenario_total:.2f}")

summary_row = outputs["summary_metrics"].iloc[0]

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Emissions Model",
    "Demand Model",
    "Revenue Model",
    "Scenario Simulator",
    "Sample Predictions"
])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Unique vehicle classes", df["Vehicle Class"].nunique())

    st.dataframe(df.head(10), use_container_width=True)

    overview_df = pd.DataFrame({
        "Task": ["Emissions", "Demand", "Revenue"],
        "Best Model": [
            summary_row["best_emissions_model"],
            summary_row["best_demand_model"],
            summary_row["best_revenue_model"]
        ]
    })
    st.subheader("Best model by task")
    st.dataframe(overview_df, use_container_width=True)

with tab2:
    results_df = outputs["emissions_results"]
    sample_df = outputs["emissions_samples"]
    importance_df = outputs["emissions_importance"]

    st.subheader("Emissions model comparison")
    st.dataframe(results_df, use_container_width=True)
    st.plotly_chart(leaderboard_chart(results_df, "Test RMSE", "Emissions model leaderboard by RMSE"), use_container_width=True)
    st.plotly_chart(model_metric_heatmap(results_df, "Emissions model metric heatmap"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(actual_vs_pred_figure(sample_df, "Emissions, actual vs predicted", "CO2"), use_container_width=True)
    with c2:
        st.plotly_chart(residual_figure(sample_df, "Emissions, residual vs predicted"), use_container_width=True)

    st.plotly_chart(residual_distribution_figure(sample_df, "Emissions, residual distribution"), use_container_width=True)

    fig = feature_importance_figure(importance_df, "Emissions feature importance")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

with tab3:
    results_df = outputs["demand_results"]
    sample_df = outputs["demand_samples"]
    importance_df = outputs["demand_importance"]

    st.subheader("Demand model comparison")
    st.dataframe(results_df, use_container_width=True)
    st.plotly_chart(leaderboard_chart(results_df, "Test RMSE", "Demand model leaderboard by RMSE"), use_container_width=True)
    st.plotly_chart(model_metric_heatmap(results_df, "Demand model metric heatmap"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(actual_vs_pred_figure(sample_df, "Demand, actual vs predicted", "Demand"), use_container_width=True)
    with c2:
        st.plotly_chart(residual_figure(sample_df, "Demand, residual vs predicted"), use_container_width=True)

    st.plotly_chart(residual_distribution_figure(sample_df, "Demand, residual distribution"), use_container_width=True)

    fig = feature_importance_figure(importance_df, "Demand feature importance")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

with tab4:
    results_df = outputs["revenue_results"]
    sample_df = outputs["revenue_samples"]
    importance_df = outputs["revenue_importance"]

    st.subheader("Revenue model comparison")
    st.dataframe(results_df, use_container_width=True)
    st.plotly_chart(leaderboard_chart(results_df, "Test RMSE", "Revenue model leaderboard by RMSE"), use_container_width=True)
    st.plotly_chart(model_metric_heatmap(results_df, "Revenue model metric heatmap"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(actual_vs_pred_figure(sample_df, "Revenue, actual vs predicted", "Revenue"), use_container_width=True)
    with c2:
        st.plotly_chart(residual_figure(sample_df, "Revenue, residual vs predicted"), use_container_width=True)

    st.plotly_chart(residual_distribution_figure(sample_df, "Revenue, residual distribution"), use_container_width=True)

    fig = feature_importance_figure(importance_df, "Revenue feature importance")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)

with tab5:
    st.subheader("Scenario simulator")

    default_scenario_df = outputs["scenario_results"]

    if abs(scenario_total - 1.0) > 1e-6:
        st.warning("Scenario shares must sum to 1.00 to run the simulator.")
        scenario_df = default_scenario_df.copy()
        current_summary = {
            "Fleet average CO2": float(scenario_df["Weighted CO2"].sum()),
            "Fleet weighted demand": float(scenario_df["Weighted Demand"].sum()),
            "Fleet weighted revenue": float(scenario_df["Weighted Revenue"].sum())
        }
    else:
        # Lightweight display-only approximation based on precomputed scenario class outputs
        base = default_scenario_df.set_index("Vehicle Class")
        scenario_mix = {
            "COMPACT": compact,
            "MID-SIZE": mid_size,
            "SUV - SMALL": small_suv,
            "SUBCOMPACT": subcompact,
            "SUV - STANDARD": standard_suv,
            "TWO-SEATER": two_seater
        }

        rows = []
        for cls, share in scenario_mix.items():
            if cls in base.index:
                row = base.loc[cls].copy()
                row["Production Share"] = share
                row["Weighted CO2"] = row["Predicted CO2 per Vehicle"] * share
                row["Weighted Demand"] = row["Predicted Demand"] * share
                row["Weighted Revenue"] = row["Predicted Revenue"] * share
                rows.append(row)

        scenario_df = pd.DataFrame(rows).reset_index().rename(columns={"index": "Vehicle Class"})
        current_summary = {
            "Fleet average CO2": float(scenario_df["Weighted CO2"].sum()),
            "Fleet weighted demand": float(scenario_df["Weighted Demand"].sum()),
            "Fleet weighted revenue": float(scenario_df["Weighted Revenue"].sum())
        }

    baseline_summary = {
        "Fleet average CO2": float(summary_row["fleet_average_co2"]),
        "Fleet weighted demand": float(summary_row["fleet_weighted_demand"]),
        "Fleet weighted revenue": float(summary_row["fleet_weighted_revenue"])
    }

    delta_co2 = current_summary["Fleet average CO2"] - baseline_summary["Fleet average CO2"]
    delta_demand = current_summary["Fleet weighted demand"] - baseline_summary["Fleet weighted demand"]
    delta_revenue = current_summary["Fleet weighted revenue"] - baseline_summary["Fleet weighted_revenue"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Fleet average CO2", f"{current_summary['Fleet average CO2']:.2f} g/km", delta=f"{delta_co2:+.2f} g/km")
    c2.metric("Fleet weighted demand", f"{current_summary['Fleet weighted demand']:.0f}", delta=f"{delta_demand:+.0f}")
    c3.metric("Fleet weighted revenue", f"£{current_summary['Fleet weighted revenue']:,.2f}", delta=f"£{delta_revenue:,.2f}")

    st.dataframe(scenario_df, use_container_width=True)
    st.plotly_chart(scenario_combo_chart(scenario_df), use_container_width=True)
    st.plotly_chart(weighted_emissions_chart(scenario_df), use_container_width=True)

with tab6:
    st.subheader("Sample predictions")
    subtab1, subtab2, subtab3 = st.tabs(["Emissions", "Demand", "Revenue"])

    with subtab1:
        st.plotly_chart(sample_predictions_figure(outputs["emissions_samples"], "Emissions sample predictions, actual vs predicted"), use_container_width=True)
        st.dataframe(outputs["emissions_samples"].head(20), use_container_width=True)

    with subtab2:
        st.plotly_chart(sample_predictions_figure(outputs["demand_samples"], "Demand sample predictions, actual vs predicted"), use_container_width=True)
        st.dataframe(outputs["demand_samples"].head(20), use_container_width=True)

    with subtab3:
        st.plotly_chart(sample_predictions_figure(outputs["revenue_samples"], "Revenue sample predictions, actual vs predicted"), use_container_width=True)
        st.dataframe(outputs["revenue_samples"].head(20), use_container_width=True)
