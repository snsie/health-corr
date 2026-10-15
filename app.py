"""Health Correlation Explorer: correlations among health measures in NHANES 2017-2018.

Run: uv run --with streamlit,pandas,seaborn,scipy,plotly streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

DATA_FILE = "nhanes_health.csv"

# Numeric columns allowed in correlations. `sex` is deliberately absent.
NUMERIC = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m²)",
    "waist_cm": "Waist circumference (cm)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "hba1c": "HbA1c (%)",
    "total_chol": "Total cholesterol (mg/dL)",
    "hdl_chol": "HDL cholesterol (mg/dL)",
    "sleep_hours": "Sleep on workdays (hours)",
    "sedentary_minutes": "Sitting per day (minutes)",
}

THRESHOLDS = {
    "None": None,
    "High blood pressure: SBP ≥ 130": ("systolic_bp", 130),
    "High blood pressure: DBP ≥ 80": ("diastolic_bp", 80),
    "Diabetes: HbA1c ≥ 6.5": ("hba1c", 6.5),
    "Obesity: BMI ≥ 30": ("bmi", 30),
    "High cholesterol: total ≥ 240": ("total_chol", 240),
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def pair_data(data: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """X and Y as columns "x" and "y", dropping missing values for this pair only.

    Generic column names keep this working when X and Y are the same variable.
    """
    return pd.DataFrame({"x": data[x], "y": data[y]}).dropna()


def pair_corr(pair: pd.DataFrame, method: str) -> float:
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan")
    return pair["x"].corr(pair["y"], method=method)


st.set_page_config(page_title="Health Correlation Explorer", layout="wide")
st.title("Health Correlation Explorer")

raw = load_data()

# ---- Sidebar: one filtered DataFrame feeds every view ---------------------
with st.sidebar:
    st.header("Filters")
    sex = st.radio("Sex", ["All", "Female", "Male"], horizontal=True)
    age_min, age_max = st.slider("Age range", 20, 80, (20, 80))
    st.header("Correlation")
    method_label = st.radio("Method", ["Pearson", "Spearman"], horizontal=True)
    st.header("Display")
    library = st.radio("Plotting library", ["seaborn", "plotly"], horizontal=True)

method = method_label.lower()
filtered = raw[raw["age"].between(age_min, age_max)]
if sex != "All":
    filtered = filtered[filtered["sex"] == sex]

st.sidebar.metric("Participants matching filters", f"{len(filtered):,} of {len(raw):,}")

# ---- 1. Data preview -------------------------------------------------------
st.header("1. The data")
st.markdown(
    "Adults aged 20 and over from the CDC's **National Health and Nutrition Examination "
    "Survey (NHANES), 2017–2018**. Each row is one participant. Blood pressure is the mean "
    "of up to three readings; sleep and sitting time are self-reported; age 80 means 80 or older. "
    "NHANES oversamples some groups and no survey weights are applied, so these correlations "
    "describe **this sample**, not the US population."
)
st.dataframe(filtered, hide_index=True, height=250)

# ---- 2. Pairwise view ------------------------------------------------------
st.header("2. Two variables")
names = list(NUMERIC)
c1, c2, c3 = st.columns(3)
x = c1.selectbox("X variable", names, index=names.index("age"), format_func=NUMERIC.get)
y = c2.selectbox("Y variable", names, index=names.index("systolic_bp"), format_func=NUMERIC.get)
color_by = c3.selectbox("Color points by", list(THRESHOLDS))

# The scatter plot always renders; only r falls back to "undefined".
plot_data = pair_data(filtered, x, y)
r, n = pair_corr(plot_data, method), len(plot_data)
m1, m2 = st.columns(2)
m1.metric(f"{method_label} r", "undefined" if pd.isna(r) else f"{r:+.2f}")
m2.metric("n (participants used)", f"{n:,}")
if pd.isna(r):
    reason = "fewer than 3 participants match" if n < 3 else "one variable has the same value for everyone shown"
    st.caption(f"r is undefined because {reason}. Widen the filters to compute it.")

hue = None
if THRESHOLDS[color_by] is not None:
    col, cutoff = THRESHOLDS[color_by]
    hue = color_by.split(": ")[1]
    plot_data[hue] = filtered.loc[plot_data.index, col].ge(cutoff).map({True: "Yes", False: "No"})

means = None
if plot_data["x"].nunique() > 10:
    bins = pd.qcut(plot_data["x"], q=10, duplicates="drop")
    means = plot_data.groupby(bins, observed=True)[["x", "y"]].mean()

if library == "seaborn":
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(
        data=plot_data, x="x", y="y", hue=hue, hue_order=["No", "Yes"] if hue else None,
        s=12, alpha=0.35, linewidth=0, ax=ax,
    )
    if means is not None:
        ax.plot(means["x"], means["y"], color="black", marker="o", linewidth=2, label="Binned mean")
        ax.legend(frameon=False)
    ax.set_xlabel(NUMERIC[x])
    ax.set_ylabel(NUMERIC[y])
    sns.despine(ax=ax)
    st.pyplot(fig)
    plt.close(fig)
else:
    fig = px.scatter(
        plot_data, x="x", y="y", color=hue, category_orders={hue: ["No", "Yes"]} if hue else None,
        opacity=0.35, labels={"x": NUMERIC[x], "y": NUMERIC[y]}, height=450,
    )
    fig.update_traces(marker_size=5)
    if means is not None:
        fig.add_trace(go.Scatter(
            x=means["x"], y=means["y"], mode="lines+markers", name="Binned mean",
            line=dict(color="black", width=2), marker=dict(color="black", size=7),
        ))
    st.plotly_chart(fig)
st.caption(
    "The black line is the mean of Y within 10 equal-sized groups of X. "
    "It can show curved patterns that a single r misses."
)

# ---- 3. Correlation heatmap ------------------------------------------------
st.header("3. Correlation heatmap")
cols = st.multiselect("Variables to include", names, default=names, format_func=NUMERIC.get)
if len(cols) < 2:
    st.info("Pick at least two variables.")
elif len(filtered) < 3:
    st.warning("Fewer than 3 participants match these filters, so no correlations can be computed. Widen the filters.")
else:
    corr = filtered[cols].corr(method=method)  # pandas drops missing values pairwise
    counts = filtered[cols].notna().astype(int)
    pair_n = counts.T @ counts
    if library == "seaborn":
        fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(cols), 0.8 + 0.7 * len(cols)))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.7, "label": f"{method_label} r"}, ax=ax,
        )
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        st.pyplot(fig)
        plt.close(fig)
    else:
        # zmin/zmax pin the color scale to -1..1, matching the seaborn version.
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            labels={"color": f"{method_label} r"}, height=150 + 55 * len(cols),
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig)
    n_lo, n_hi = int(pair_n.values.min()), int(pair_n.values.max())
    n_text = f"n = {n_lo:,}" if n_lo == n_hi else f"n = {n_lo:,} to {n_hi:,} depending on the pair"
    st.caption(f"{method_label} correlation, {n_text}. Colors are fixed from −1 to 1, so weak correlations look pale.")
