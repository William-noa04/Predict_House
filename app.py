import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
from pathlib import Path
import keras

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KC House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load logo as base64 ───────────────────────────────────────────────────────
def get_logo_base64():
    for ext in ["logo.jpeg", "logo.jpg", "logo.png"]:
        logo_path = Path(ext)
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                mime = "png" if ext.endswith("png") else "jpeg"
                return base64.b64encode(f.read()).decode(), mime
    return None, None

logo_b64, logo_mime = get_logo_base64()
logo_html = (
    f'<img src="data:image/{logo_mime};base64,{logo_b64}" style="height:70px;object-fit:contain;">'
    if logo_b64
    else '<div style="font-size:2rem;font-weight:900;color:#C9A84C;">KC</div>'
)

# ── Gold/White CSS theme ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --gold-dark:   #9A7B2F;
        --gold:        #C9A84C;
        --gold-light:  #E8CC7A;
        --gold-pale:   #FDF6E3;
        --white:       #FFFFFF;
        --grey-400:    #AAAAAA;
        --dark:        #1C1C1C;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--white);
        color: var(--dark);
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1C1C1C 0%, #2E2A1E 100%);
        border-right: 2px solid var(--gold);
    }
    [data-testid="stSidebar"] * { color: var(--gold-light) !important; }
    [data-testid="stSidebar"] label { color: var(--gold-pale) !important; font-weight: 500; }
    [data-testid="stSidebar"] .stSlider > div > div > div { background: var(--gold) !important; }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #2C2714 !important;
        border-color: var(--gold-dark) !important;
    }

    /* ── Top header bar ── */
    .top-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, #1C1C1C, #3A2F10);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--gold);
    }
    .top-bar-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        color: var(--white);
        letter-spacing: 0.03em;
    }
    .top-bar-subtitle { color: var(--gold-light); font-size: 0.85rem; margin-top: 2px; }

    /* ── Metric cards ── */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 160px;
        background: var(--white);
        border: 1.5px solid var(--gold-light);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        box-shadow: 0 2px 12px rgba(201,168,76,0.12);
        transition: transform .2s, box-shadow .2s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(201,168,76,0.22); }
    .metric-card .mc-label { font-size: 0.78rem; color: var(--gold-dark); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
    .metric-card .mc-value { font-family: 'Playfair Display', serif; font-size: 1.7rem; color: var(--dark); margin-top: 4px; }
    .metric-card .mc-icon  { font-size: 1.3rem; float: right; margin-top: -2.2rem; opacity: .8; }

    /* ── Section title ── */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        color: var(--gold-dark);
        border-left: 4px solid var(--gold);
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem;
    }

    /* ── Result card ── */
    .result-card {
        background: linear-gradient(135deg, #1C1C1C 0%, #3A2F10 100%);
        border: 2px solid var(--gold);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(201,168,76,0.25);
        margin-top: 1rem;
    }
    .result-card .res-label {
        color: var(--gold-light);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: .1em;
        margin-bottom: 0.5rem;
    }
    .result-card .res-price {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: var(--gold-light);
        text-shadow: 0 2px 12px rgba(201,168,76,0.5);
    }
    .result-card .res-note {
        color: var(--gold-pale);
        font-size: 0.8rem;
        margin-top: 0.6rem;
        opacity: .75;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--gold-dark), var(--gold-light));
        color: #1C1C1C !important;
        font-weight: 700;
        font-family: 'DM Sans', sans-serif;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2.5rem;
        font-size: 1rem;
        letter-spacing: .05em;
        transition: all .25s;
        box-shadow: 0 4px 14px rgba(201,168,76,0.35);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(201,168,76,0.5);
        background: linear-gradient(135deg, var(--gold), var(--gold-light));
    }

    /* ── Inputs ── */
    .stSlider > div > div > div { background: var(--gold) !important; }
    .stNumberInput input, .stSelectbox > div > div {
        border-color: var(--gold-light) !important;
        border-radius: 8px !important;
    }

    hr { border-color: var(--gold-light) !important; opacity: .35; }
    .stAlert { border-left-color: var(--gold) !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .stTabs [aria-selected="true"] {
        color: var(--gold-dark) !important;
        border-bottom-color: var(--gold) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model & scalers ──────────────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def load_assets():
    errors = []
    model, scaler_x, scaler_y = None, None, None

    for model_file in ["rf_model.keras", "rf_model.h5"]:
        if Path(model_file).exists():
            try:
                model = keras.models.load_model(model_file)
                break
            except Exception as e:
                errors.append(f"Modèle ({model_file}) : {e}")

    if model is None:
        errors.append("rf_model.keras introuvable")

    if Path("scaler_x.joblib").exists():
        try:
            scaler_x = joblib.load("scaler_x.joblib")
        except Exception as e:
            errors.append(f"scaler_x : {e}")
    else:
        errors.append("scaler_x.joblib introuvable")

    if Path("scaler_y.joblib").exists():
        try:
            scaler_y = joblib.load("scaler_y.joblib")
        except Exception as e:
            errors.append(f"scaler_y : {e}")
    else:
        errors.append("scaler_y.joblib introuvable")

    return model, scaler_x, scaler_y, errors
    errors = []
    model, scaler_x, scaler_y = None, None, None

    # Keras model
    for model_file in ["rf_model.keras", "rf_model.h5"]:
        if Path(model_file).exists():
            try:
                model = tf.keras.models.load_model(model_file)
                break
            except Exception as e:
                errors.append(f"Modèle ({model_file}) : {e}")

    if model is None and not any("Modèle" in e for e in errors):
        errors.append("Modèle : rf_model.keras introuvable")

    # Scalers
    if Path("scaler_x.joblib").exists():
        try:
            scaler_x = joblib.load("scaler_x.joblib")
        except Exception as e:
            errors.append(f"scaler_x : {e}")
    else:
        errors.append("scaler_x.joblib introuvable")

    if Path("scaler_y.joblib").exists():
        try:
            scaler_y = joblib.load("scaler_y.joblib")
        except Exception as e:
            errors.append(f"scaler_y : {e}")
    else:
        errors.append("scaler_y.joblib introuvable")

    return model, scaler_x, scaler_y, errors

model, scaler_x, scaler_y, load_errors = load_assets()

# ── Prediction function ───────────────────────────────────────────────────────
def predire_prix_maison(GrLivArea, BedroomAbvGr, FullBath):
    nouvelle_maison = np.array([[GrLivArea, BedroomAbvGr, FullBath]])
    nouvelle_maison_scaled = scaler_x.transform(nouvelle_maison)
    prediction_scaled = model.predict(nouvelle_maison_scaled, verbose=0)
    prix_reel = scaler_y.inverse_transform(prediction_scaled)
    return round(float(prix_reel[0][0]), 2)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="padding:1.5rem 1rem 0.5rem;text-align:center;border-bottom:1px solid #9A7B2F;margin-bottom:1.2rem;">'
        f'{logo_html}'
        f'<p style="font-family:Playfair Display,serif;font-size:1rem;margin-top:.5rem;color:#E8CC7A;">House Price Predictor</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📐 Paramètres de la maison")

    GrLivArea    = st.number_input(
        "📐 Surface habitable (sqft)",
        min_value=500, max_value=10000, value=1800, step=50,
        help="GrLivArea : surface habitable au-dessus du sol en pieds carrés"
    )
    BedroomAbvGr = st.slider(
        "🛏  Nombre de chambres", 0, 10, 3,
        help="BedroomAbvGr : chambres au-dessus du sol"
    )
    FullBath     = st.slider(
        "🚿  Salles de bain complètes", 0, 5, 2,
        help="FullBath : salles de bain avec baignoire ou douche"
    )

    st.markdown("---")
    predict_btn = st.button("🏠  Prédire le Prix")

    # Status des fichiers
    st.markdown("---")
    st.markdown("**🗂 État des fichiers**")
    st.markdown(f"{'✅' if model    else '❌'} `rf_model.keras`")
    st.markdown(f"{'✅' if scaler_x else '❌'} `scaler_x.joblib`")
    st.markdown(f"{'✅' if scaler_y else '❌'} `scaler_y.joblib`")

# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="top-bar">
        <div>
            <div class="top-bar-title">KC — House Price Predictor</div>
            <div class="top-bar-subtitle">Réseau de Neurones Keras · Dataset King County House Prices</div>
        </div>
        <div>{logo_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load error banners ────────────────────────────────────────────────────────
if load_errors:
    for err in load_errors:
        st.warning(f"⚠️ {err}")

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="metric-row">
        <div class="metric-card">
            <div class="mc-label">Propriétés dataset</div>
            <div class="mc-value">1 460</div>
            <div class="mc-icon">🏘</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Variables utilisées</div>
            <div class="mc-value">3</div>
            <div class="mc-icon">📊</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Architecture</div>
            <div class="mc-value">64 → 32 → 1</div>
            <div class="mc-icon">🤖</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Optimiseur</div>
            <div class="mc-value">Adam · MSE</div>
            <div class="mc-icon">⚡</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏠 Prédiction", "📊 Aperçu Dataset"])

# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Paramètres saisis</div>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Paramètre":   [
                "Surface habitable (GrLivArea)",
                "Chambres (BedroomAbvGr)",
                "Salles de bain (FullBath)",
            ],
            "Valeur": [
                f"{GrLivArea:,} sqft",
                str(BedroomAbvGr),
                str(FullBath),
            ],
            "Description": [
                "Surface en sqft au-dessus du sol",
                "Nb de chambres hors sous-sol",
                "Salles de bain avec baignoire/douche",
            ],
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">Architecture du réseau</div>', unsafe_allow_html=True)
        st.markdown(
            """
            | Couche  | Neurones | Activation |
            |---------|----------|------------|
            | Dense 1 | 64       | ReLU       |
            | Dense 2 | 32       | ReLU       |
            | Sortie  | 1        | Linéaire   |

            > **Loss** : Mean Squared Error &nbsp;·&nbsp; **Métrique** : MAE &nbsp;·&nbsp; **Epochs** : 100
            """
        )

    with col_right:
        st.markdown('<div class="section-title">Résultat</div>', unsafe_allow_html=True)

        if predict_btn:
            if not all([model, scaler_x, scaler_y]):
                st.error(
                    "❌ Fichiers manquants. Assurez-vous que `rf_model.keras`, "
                    "`scaler_x.joblib` et `scaler_y.joblib` sont dans le même dossier que `app.py`."
                )
            else:
                try:
                    with st.spinner("Calcul en cours..."):
                        price = predire_prix_maison(GrLivArea, BedroomAbvGr, FullBath)

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="res-label">Prix estimé de la propriété</div>
                            <div class="res-price">${price:,.0f}</div>
                            <div class="res-note">
                                {GrLivArea:,} sqft · {BedroomAbvGr} chambre(s) · {FullBath} salle(s) de bain
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Fourchette ±10 %
                    low, high = price * 0.90, price * 1.10
                    st.markdown('<div class="section-title">Fourchette estimée (±10 %)</div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Min",    f"${low:,.0f}")
                    c2.metric("Estimé", f"${price:,.0f}", delta="référence")
                    c3.metric("Max",    f"${high:,.0f}")

                    # Comparaison marché
                    st.markdown('<div class="section-title">Comparaison au marché</div>', unsafe_allow_html=True)
                    chart_df = pd.DataFrame({
                        "Bien":    ["Médiane marché", "Votre bien"],
                        "Prix ($)": [163_000, price],
                    }).set_index("Bien")
                    st.bar_chart(chart_df, color="#C9A84C")

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")

        else:
            st.info("👈 Ajustez les paramètres dans la barre latérale puis cliquez sur **Prédire le Prix**.")

            # Exemples rapides
            if all([model, scaler_x, scaler_y]):
                st.markdown('<div class="section-title">Exemples rapides</div>', unsafe_allow_html=True)
                exemples = [
                    ("🏠 Petit appartement", 1200, 2, 1),
                    ("🏡 Maison standard",   2000, 3, 2),
                    ("🏰 Grande villa",      2500, 4, 3),
                ]
                cols = st.columns(3)
                for col, (label, sqft, bed, bath) in zip(cols, exemples):
                    try:
                        p = predire_prix_maison(sqft, bed, bath)
                        col.metric(label, f"${p:,.0f}", f"{sqft} sqft · {bed}ch · {bath}sdb")
                    except Exception:
                        col.info("—")

# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Aperçu — train.csv</div>', unsafe_allow_html=True)

    try:
        df_raw = pd.read_csv("train.csv")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Lignes totales", f"{len(df_raw):,}")
        col_b.metric("Prix moyen",     f"${df_raw['SalePrice'].mean():,.0f}")
        col_c.metric("Prix médian",    f"${df_raw['SalePrice'].median():,.0f}")
        col_d.metric("Prix max",       f"${df_raw['SalePrice'].max():,.0f}")

        st.markdown('<div class="section-title">Données (20 premières lignes)</div>', unsafe_allow_html=True)
        cols_show = ["Id", "GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]
        available = [c for c in cols_show if c in df_raw.columns]
        st.dataframe(df_raw[available].head(20), use_container_width=True)

        st.markdown('<div class="section-title">Distribution de SalePrice</div>', unsafe_allow_html=True)
        hist = df_raw["SalePrice"].value_counts(bins=30).sort_index().rename("Nb propriétés")
        st.bar_chart(hist, color="#C9A84C")

    except FileNotFoundError:
        st.warning("⚠️ `train.csv` introuvable. Placez-le dans le même répertoire que `app.py`.")
        st.dataframe(
            pd.DataFrame({
                "GrLivArea":    [1200, 2000, 2500, 1500, 3000],
                "BedroomAbvGr": [2, 3, 4, 2, 5],
                "FullBath":     [1, 2, 3, 1, 3],
                "SalePrice":    [120000, 200000, 280000, 150000, 350000],
            }),
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr>
    <div style="text-align:center;color:#AAAAAA;font-size:.78rem;padding:.5rem 0 1rem;">
        KC House Price Predictor · Réseau de Neurones Keras · Dataset King County · 2024
    </div>
    """,
    unsafe_allow_html=True,
)