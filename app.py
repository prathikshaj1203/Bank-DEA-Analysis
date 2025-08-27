# app.py
# ---------------------------------------
# Streamlit Web App for DEA Bank Analysis
# ---------------------------------------
# How to run:
#   1) pip install streamlit plotly
#   2) streamlit run app.py
# ---------------------------------------

import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Your project modules ---
from src.data_preprocessing import load_data, clean_data
from src.dea_analysis import dea_ccr, dea_bcc
from src.reporting import top_efficient_banks

# ---------- Page setup ----------
st.set_page_config(
    page_title="Bank DEA Efficiency Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ---------- Helper / Cache ----------
@st.cache_data(show_spinner=True)
def prepare_results(data_path: str) -> pd.DataFrame:
    """
    Load, clean, compute CCR/BCC DEA and Scale efficiency, and return results dataframe.
    Cached for faster reruns.
    """
    df = load_data(data_path)
    df = clean_data(df)

    # Define Inputs / Outputs (must match clean_data column names)
    input_cols = [
        "employees",
        "branches",
        "atms/outlets",
        "interest_expense_(cr)",
        "operating_expenses_(cr)",
        "total_deposits_(cr)"
    ]
    output_cols = [
        "total_advances_(cr)",
        "non-interest_income_(cr)",
        "net_profit_(cr)",
        "roa_(%)",
        "roe_(%)",
        "adjusted_net_profit"
    ]

    X = df[input_cols]
    Y = df[output_cols]

    eff_ccr = dea_ccr(X, Y)
    eff_bcc = dea_bcc(X, Y)

    # Avoid division by zero for scale efficiency
    safe_bcc = np.where(eff_bcc == 0, np.nan, eff_bcc)
    scale_eff = eff_ccr / safe_bcc

    results = df.copy()
    results["efficiency_ccr"] = eff_ccr
    results["efficiency_bcc"] = eff_bcc
    results["scale_efficiency"] = scale_eff

    # Sort for consistent displays
    results = results.sort_values(["year", "bank_name"]).reset_index(drop=True)
    return results


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_heatmap(df: pd.DataFrame, metric: str) -> go.Figure:
    pivot_df = df.pivot(index="bank_name", columns="year", values=metric)
    fig = px.imshow(
        pivot_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"{metric.upper()} Heatmap (Banks × Years)"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), height=750)
    return fig


def make_year_dropdown_bars(df: pd.DataFrame, metric: str) -> go.Figure:
    years = sorted(df["year"].unique())
    fig = go.Figure()

    for yr in years:
        df_year = df[df["year"] == yr]
        fig.add_trace(go.Bar(
            x=df_year["bank_name"],
            y=df_year[metric],
            name=str(yr),
            visible=(yr == years[0])
        ))

    buttons = []
    for i, yr in enumerate(years):
        visible = [False] * len(years)
        visible[i] = True
        buttons.append(dict(
            label=str(yr),
            method="update",
            args=[{"visible": visible},
                  {"title": f"{metric.upper()} Scores - {yr}"}]
        ))

    fig.update_layout(
        title=f"{metric.upper()} Scores - {years[0]}",
        xaxis_title="Bank",
        yaxis_title="Efficiency Score",
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 1.02,
            "y": 1.1
        }],
        legend_title="Year",
        height=600,
        margin=dict(l=40, r=40, t=80, b=80),
        xaxis_tickangle=-45
    )
    return fig


def make_trends(df: pd.DataFrame, metric: str, selected_banks: list[str]) -> go.Figure:
    if selected_banks:
        plot_df = df[df["bank_name"].isin(selected_banks)]
        title_suffix = " (Selected Banks)"
    else:
        plot_df = df
        title_suffix = " (All Banks)"

    fig = px.line(
        plot_df,
        x="year",
        y=metric,
        color="bank_name",
        markers=True,
        title=f"{metric.upper()} Trends 2019–2024{title_suffix}"
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(dtick=1)
    )
    return fig


def make_top_banks(df: pd.DataFrame, metric: str, year: int, top_n: int) -> go.Figure:
    df_year = df[df["year"] == year].sort_values(by=metric, ascending=False).head(top_n)
    fig = px.bar(
        df_year,
        x="bank_name",
        y=metric,
        text=metric,
        title=f"Top {top_n} Banks in {year} ({metric.upper()})"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=60, b=80),
        xaxis_tickangle=-45
    )
    return fig


# ---------- Sidebar Controls ----------
st.sidebar.title("⚙️ Controls")

data_path = st.sidebar.text_input(
    "Excel file path",
    value="data/Bank_Analysis.xlsx",
    help="Path to your cleaned dataset file."
)

metric = st.sidebar.selectbox(
    "Efficiency Metric",
    ["efficiency_ccr", "efficiency_bcc", "scale_efficiency"],
    index=0
)

# Load data & compute results
try:
    df_results = prepare_results(data_path)
except Exception as e:
    st.error(f"Error loading or processing data. Check the path and format.\n\n{e}")
    st.stop()

# Additional sidebar filters
all_years = sorted(df_results["year"].unique())
year_for_top = st.sidebar.selectbox("Year for 'Top Banks' chart", all_years, index=len(all_years) - 1)
top_n = st.sidebar.slider("Top N Banks", min_value=5, max_value=20, value=10, step=1)

all_banks = sorted(df_results["bank_name"].unique())
selected_banks = st.sidebar.multiselect(
    "Filter specific banks for trend chart (optional):",
    options=all_banks,
    default=[]
)

# ---------- Header ----------
st.title("🏦 Bank Efficiency Analysis (DEA Dashboard)")
st.caption("CCR, BCC, and Scale efficiency across 33 banks for years 2019–2024.")

# ---------- Metrics Overview ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Banks", f"{df_results['bank_name'].nunique()}")
with col2:
    st.metric("Years", f"{df_results['year'].nunique()}")
with col3:
    st.metric("Rows", f"{len(df_results):,}")
with col4:
    latest_year = max(all_years)
    latest_mean = df_results[df_results["year"] == latest_year][metric].mean()
    st.metric(f"Avg {metric.upper()} ({latest_year})", f"{latest_mean:.3f}")

st.divider()

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Data",
    "📊 Year-wise (Dropdown)",
    "📈 Trends",
    "🗺️ Heatmap",
    "🏆 Top Banks"
])

with tab1:
    st.subheader("Dataset (with DEA results)")
    st.dataframe(df_results, use_container_width=True, height=500)

    st.download_button(
        "Download results CSV",
        data=df_to_csv_bytes(df_results),
        file_name="efficiency_scores.csv",
        mime="text/csv",
        use_container_width=True
    )

with tab2:
    st.subheader("Year-wise Efficiency (choose year from dropdown on the chart)")
    fig_year = make_year_dropdown_bars(df_results, metric)
    st.plotly_chart(fig_year, use_container_width=True)

with tab3:
    st.subheader("Efficiency Trends by Bank")
    st.caption("Use the sidebar to filter specific banks (or leave empty to show all).")
    fig_trend = make_trends(df_results, metric, selected_banks)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab4:
    st.subheader("Heatmap: Banks × Years")
    fig_heat = make_heatmap(df_results, metric)
    st.plotly_chart(fig_heat, use_container_width=True)

with tab5:
    st.subheader("Top Efficient Banks")
    fig_top = make_top_banks(df_results, metric, year_for_top, top_n)
    st.plotly_chart(fig_top, use_container_width=True)

    st.write(f"**Top {top_n} in {year_for_top} ({metric.upper()})**")
    top_df = df_results[df_results["year"] == year_for_top][["bank_name", metric]] \
        .sort_values(by=metric, ascending=False) \
        .head(top_n) \
        .reset_index(drop=True)
    st.dataframe(top_df, use_container_width=True)

    st.download_button(
        f"Download Top {top_n} ({year_for_top})",
        data=df_to_csv_bytes(top_df),
        file_name=f"top_{top_n}_banks_{year_for_top}_{metric}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with Streamlit + Plotly • DEA models: CCR, BCC • Scale = CCR/BCC")

from src.clustering import cluster_banks

# ---------- Clustering Tab ----------
st.header("🌀 Clustering Analysis")

n_clusters = st.slider("Number of Clusters", 2, 6, 3)
cluster_features = ["efficiency_ccr", "efficiency_bcc", "scale_efficiency"]

clustered_df, _ = cluster_banks(df_results, cluster_features, n_clusters)

fig_cluster = px.scatter(
    clustered_df, x="pca1", y="pca2", color="cluster",
    hover_data=["bank_name", "year"],
    title="Clusters of Banks (PCA 2D)"
)

st.plotly_chart(fig_cluster, use_container_width=True)
st.dataframe(clustered_df[["bank_name", "year", "cluster"]])
