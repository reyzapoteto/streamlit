import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# ============================
# CONFIGURATION & STYLING
# ============================

st.set_page_config(
    page_title="Real Estate Analytics",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# DESIGN SYSTEM & CSS
# ============================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;700&display=swap');

    :root {
        --primary: #2563EB;
        --primary-dark: #1E40AF;
        --secondary: #64748B;
        --background: #F8FAFC;
        --surface: #FFFFFF;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --text-main: #1E293B;
        --text-sub: #64748B;
        --border: #E2E8F0;
    }

    /* General Reset */
    .stApp {
        background-color: var(--background);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: var(--text-main);
        font-weight: 700;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Custom Cards */
    .metric-card {
        background: var(--surface);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid var(--border);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-sub);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }

    .metric-suffix {
        font-size: 1rem;
        color: var(--text-sub);
        font-weight: 500;
        margin-left: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }

    /* Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: var(--surface);
        border-radius: 12px;
        border: 1px solid var(--border);
        color: var(--text-main);
    }

    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {
        border-color: var(--primary);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: var(--text-sub);
        font-weight: 600;
        padding-bottom: 12px;
        font-family: 'Inter', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }

    /* Plotly Chart Container */
    .chart-box {
        background: var(--surface);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }

    /* DataFrame Styling */
    .dataframe {
        font-family: 'Inter', sans-serif !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    /* Sidebar Navigation Customization */
    /* Target the option_menu container if possible via CSS, primarily done in Python args, but here for backup */

    </style>
    """,
    unsafe_allow_html=True,
)

# Define Colors for Python usage matching CSS
COLORS = {
    "PRIMARY": "#2563EB",
    "PRIMARY_DARK": "#1E40AF",
    "BG_LIGHT": "#F8FAFC",
    "CARD": "#FFFFFF",
    "TEXT_MAIN": "#1E293B",
    "TEXT_SUB": "#64748B",
    "SUCCESS": "#10B981",
    "CHART_SEQ": [
        "#2563EB",  # Blue
        "#10B981",  # Emerald
        "#F59E0B",  # Amber
        "#EF4444",  # Red
        "#8B5CF6",  # Violet
        "#EC4899",  # Pink
        "#06B6D4",  # Cyan
        "#6366F1",  # Indigo
    ],
}

# ============================
# DATA HANDLING
# ============================


@st.cache_resource
def load_model():
    """Load model pipeline XGBoost"""
    return joblib.load("model_prediksi_xgboost.sav")


@st.cache_data
def load_and_clean_data():
    """Load and perform robust cleaning on the dataset for analysis"""
    df = pd.read_csv("rumah123.csv", on_bad_lines="skip")

    # --- SAFETY CHECK ---
    required_cols = [
        "harga",
        "luas_tanah",
        "luas_bangunan",
        "daya_watt",
        "kamar_tidur",
        "kamar_mandi",
        "jumlah_garasi",
        "jumlah_lantai",
        "kota_original",
        "sertifikat_rumah_original",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
        st.stop()

    return df


def get_model_features():
    """
    Define the exact features expected by the model based on training data.
    """
    return [
        "kamar_tidur",
        "kamar_mandi",
        "luas_tanah",
        "luas_bangunan",
        "jumlah_garasi",
        "daya_watt",
        "jumlah_lantai",
        "sertifikat_Hak Pakai",
        "sertifikat_Lainnya",
        "sertifikat_PPJB",
        "sertifikat_SHM",
        "kota_Batu",
        "kota_Gresik",
        "kota_Jember",
        "kota_Kediri",
        "kota_Lumajang",
        "kota_Madiun",
        "kota_Malang",
        "kota_Mojokerto",
        "kota_Nganjuk",
        "kota_Pamekasan",
        "kota_Pasuruan",
        "kota_Probolinggo",
        "kota_Sidoarjo",
        "kota_Situbondo",
        "kota_Surabaya",
        "kota_Trenggalek",
        "kota_Tuban",
        "kota_Tulungagung",
    ]



# Load data and feature structure
df = load_and_clean_data()
model_features = get_model_features()
model = load_model()

# ============================
# UTILS
# ============================


def format_idr(value):
    """Format number to IDR currency string"""
    if value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.1f} Miliar"
    elif value >= 1_000_000:
        return f"Rp {value/1_000_000:.0f} Juta"
    else:
        return f"Rp {value:,.0f}"


def card_metric(label, value, suffix="", icon=None):
    icon_html = (
        f'<div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>'
        if icon
        else ""
    )
    st.markdown(
        f"""
        <div class="metric-card">
            {icon_html}
            <div class="metric-label">{label}</div>
            <div class="metric-value">
                {value}<span class="metric-suffix">{suffix}</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


# ============================
# PAGE: DASHBOARD (ANALYSIS)
# ============================


def page_dashboard():
    st.title("Property Market Analysis")
    st.markdown("Deep dive into the property data insights and trends.")

    # Top Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card_metric("Properties", f"{len(df):,}", icon="🏠")
    with c2:
        card_metric("Avg Price", format_idr(df["harga"].mean()), icon="🏷️")
    with c3:
        card_metric("Avg Land", f"{df['luas_tanah'].mean():.0f}", "m²", icon="📐")
    with c4:
        card_metric("Top Location", df["kota_original"].mode()[0], icon="📍")

    st.markdown("---")

    # Row 1: Distribution & Cities
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.subheader("🏙️ Property Distribution by City")

        # Aggregate top cities
        city_counts = df["kota_original"].value_counts().reset_index()
        city_counts.columns = ["kota_original", "Count"]
        top_cities = city_counts.head(10)

        fig_city = go.Figure(
            go.Bar(
                x=top_cities["Count"],
                y=top_cities["kota_original"],
                orientation="h",
                text=top_cities["Count"],
                textposition="outside",
                textfont=dict(size=12, family="Inter", color=COLORS["TEXT_SUB"]),
                marker=dict(
                    color=top_cities["Count"],
                    colorscale="Blues",
                    line=dict(width=0),
                    cornerradius=4,
                ),
                hovertemplate="<b>%{y}</b><br>Properties: %{x}<extra></extra>",
            )
        )
        fig_city.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=400,
            margin=dict(l=0, r=40, t=10, b=10),
            font=dict(color=COLORS["TEXT_MAIN"], family="Inter"),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
                title="Number of Properties",
                title_font=dict(size=12),
            ),
            yaxis=dict(showgrid=False, title="", autorange="reversed"),
        )
        st.plotly_chart(fig_city, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.subheader("💰 Price Distribution")

        # Filter outliers: remove negative/zero prices and cap at 99th percentile
        price_99th = df["harga"].quantile(0.99)
        df_price_filtered = df[(df["harga"] > 0) & (df["harga"] <= price_99th)]

        # Calculate histogram manually to enable color-by-count
        counts, bins = np.histogram(df_price_filtered["harga"], bins=40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        fig_hist = go.Figure(
            go.Bar(
                x=bin_centers,
                y=counts,
                marker=dict(
                    color=counts,
                    colorscale="Blues",
                    line=dict(color="white", width=0),
                ),
                hovertemplate="Price: %{x}<br>Count: %{y}<extra></extra>",
            )
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Price (IDR)",
                title_font=dict(size=12),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
                tickformat=",.0f",
            ),
            yaxis=dict(
                title="Count",
                title_font=dict(size=12),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            font=dict(color=COLORS["TEXT_MAIN"], family="Inter"),
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Features
    st.markdown("### Property Features Analysis")

    # Use HTML container for background
    with st.container():
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(
            ["Correlation Analysis", "Price vs Area", "Certificate Types"]
        )

        with tab1:
            # Correlation Heatmap
            corr_cols = [
                "harga",
                "luas_tanah",
                "luas_bangunan",
                "kamar_tidur",
                "kamar_mandi",
                "jumlah_garasi",
            ]
            corr = df[corr_cols].corr()

            # Rename columns for better display
            display_names = {
                "harga": "Price",
                "luas_tanah": "Land Area",
                "luas_bangunan": "Building Area",
                "kamar_tidur": "Bedrooms",
                "kamar_mandi": "Bathrooms",
                "jumlah_garasi": "Garage",
            }
            corr_display = corr.rename(index=display_names, columns=display_names)

            fig_corr = go.Figure(
                go.Heatmap(
                    z=corr_display.values,
                    x=corr_display.columns.tolist(),
                    y=corr_display.index.tolist(),
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(corr_display.values, 2),
                    texttemplate="%{text}",
                    textfont=dict(size=12, family="Inter"),
                    hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>",
                )
            )
            fig_corr.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["TEXT_MAIN"], family="Inter"),
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(side="bottom", tickangle=0),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab2:
            # Scatter Price vs Land
            plot_df = df.copy()

            fig_scatter = px.scatter(
                plot_df,
                x="luas_tanah",
                y="harga",
                color="kota_original",
                size="kamar_tidur",
                hover_data=["kota_original"],
                # Use a cleaner color sequence
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_scatter.update_traces(
                marker=dict(opacity=0.6, line=dict(width=0.5, color="white"), size=6),
                hovertemplate="<b>%{customdata[0]}</b><br>Land: %{x:,.0f} m²<br>Price: Rp %{y:,.0f}<extra></extra>",
            )
            fig_scatter.update_layout(
                xaxis=dict(
                    title="Land Area (m²)",
                    title_font=dict(size=12),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.05)",
                ),
                yaxis=dict(
                    title="Price (IDR)",
                    title_font=dict(size=12),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.05)",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=500,
                font=dict(color=COLORS["TEXT_MAIN"], family="Inter"),
                showlegend=False,  # Hide legend to reduce clutter
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            # Certificate Pie/Bar
            sert_counts = df["sertifikat_rumah_original"].value_counts().reset_index()
            sert_counts.columns = ["Type", "Count"]

            total_count = sert_counts["Count"].sum()
            fig_pie = go.Figure(
                go.Pie(
                    labels=sert_counts["Type"],
                    values=sert_counts["Count"],
                    hole=0.6,
                    marker=dict(
                        colors=["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"],
                        line=dict(color="white", width=2),
                    ),
                    # Hide labels for slices < 2% to prevent overlap
                    text=[
                        f"{l}<br>{v/total_count:.1%}" if v / total_count > 0.02 else ""
                        for l, v in zip(sert_counts["Type"], sert_counts["Count"])
                    ],
                    textinfo="text",
                    textposition="outside",
                    textfont=dict(size=12, family="Inter"),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                    pull=[0.02] * len(sert_counts),
                )
            )

            # Add center annotation
            fig_pie.add_annotation(
                text=f"<b>{sert_counts['Count'].sum():,}</b><br>Total",
                x=0.5,
                y=0.5,
                font=dict(size=20, color=COLORS["PRIMARY_DARK"], family="Outfit"),
                showarrow=False,
            )

            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["TEXT_MAIN"], family="Inter"),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
                height=450,
                margin=dict(l=20, r=20, t=30, b=50),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ============================
# PAGE: PREDICTION
# ============================


def page_prediction():
    st.markdown(
        '<h1 style="margin-bottom: 1rem;">🔮 Price Estimator</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter property details below to estimate its market based on our *XGBoost AI Model*."
    )

    # Input Form Container
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.subheader("Property Details")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    kota_options = sorted(df["kota_original"].unique().tolist())
    sertifikat_options = sorted(df["sertifikat_rumah_original"].dropna().unique().tolist())

    with col1:
        kota = st.selectbox("City / Location", kota_options)
        kamar_tidur = st.number_input("Bedrooms", 0, 20, 3)
        kamar_mandi = st.number_input("Bathrooms", 0, 20, 2)
        jumlah_lantai = st.number_input("Floors", 1, 10, 1)

    with col2:
        luas_tanah = st.number_input("Land Area (m²)", 10, 5000, 100)
        luas_bangunan = st.number_input("Building Area (m²)", 10, 5000, 90)
        jumlah_garasi = st.number_input("Garage Capacity", 0, 10, 1)
        daya_listrik = st.number_input("Electricity (Watt)", 450, 50000, 1300)

    sertifikat = st.selectbox("Certificate Type", sertifikat_options)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Calculate Estimate"):
        try:
            # 1. Create a base dictionary with 0s for all model features
            input_data = {col: [0] for col in model_features}
            input_df = pd.DataFrame(input_data)

            # 2. Fill Numeric Values
            input_df["kamar_tidur"] = kamar_tidur
            input_df["kamar_mandi"] = kamar_mandi
            input_df["luas_tanah"] = luas_tanah
            input_df["luas_bangunan"] = luas_bangunan
            input_df["jumlah_garasi"] = jumlah_garasi
            input_df["jumlah_lantai"] = jumlah_lantai
            input_df["daya_watt"] = daya_listrik

            # 3. Set Categorical (One-Hot)
            # kota_Surabaya -> 1
            kota_col = f"kota_{kota}"
            if kota_col in input_df.columns:
                input_df[kota_col] = 1

            # sertifikat_rumah_SHM -> 1 (Note: App uses 'sertifikat_' prefix match model)
            sert_col = f"sertifikat_{sertifikat}"
            if sert_col in input_df.columns:
                input_df[sert_col] = 1

            # 4. Predict
            input_df = input_df[model_features]
            pred = model.predict(input_df)[0]

            # Result
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
                    padding: 32px;
                    border-radius: 20px;
                    border: 1px solid #10B981;
                    margin-top: 32px;
                    text-align: center;
                    box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1), 0 2px 4px -1px rgba(16, 185, 129, 0.06);
                ">
                    <h4 style="color: #047857; margin:0; font-weight: 600; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.05em;">Estimated Market Price</h4>
                    <h1 style="
                        background: linear-gradient(to right, #059669, #10B981);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-size: 3rem;
                        margin: 16px 0;
                        font-weight: 800;
                        font-family: 'Outfit', sans-serif;
                    ">{format_idr(pred)}</h1>
                    <p style="color: #059669; font-size: 0.95rem; opacity: 0.9;">
                        Based on XGBoost Algorithm dengan Hasil Evaluasi 0,71 • Location: {kota}
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.error(
                "Tip: Try selecting a city/certificate combination that exists in the dataset."
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================
# MAIN APP NAVIGATION
# ============================


def main():
    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2.5rem; margin-bottom: 0;">🏠</h1>
                <h2 style="font-family: 'Outfit'; font-size: 1.5rem; color: var(--primary-dark);">RumahData</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Option Menu for Navigation
        selected = option_menu(
            None,
            ["Stats & Analysis", "Price Prediction"],
            icons=["bar-chart-line-fill", "calculator-fill"],
            menu_icon="cast",
            default_index=1,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {"font-size": "1.1rem"},
                "nav-link": {
                    "font-size": "1rem",
                    "text-align": "left",
                    "margin": "0.5rem 0",
                    "border-radius": "12px",
                    "padding": "0.75rem 1rem",
                    "color": COLORS["TEXT_SUB"],
                    "font-weight": "500",
                    "border": "1px solid transparent",
                    "font-family": "Inter",
                },
                "nav-link-selected": {
                    "background-color": COLORS["PRIMARY"],
                    "color": "white",
                    "box-shadow": "0 4px 6px -1px rgba(37, 99, 235, 0.2)",
                    "font-weight": "600",
                },
            },
        )

        st.markdown("---")

    # Routing
    if selected == "Stats & Analysis":
        page_dashboard()
    else:
        page_prediction()


if _name_ == "_main_":
    main()
