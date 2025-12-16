import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# Necesario para reconstruir pipelines
from preprocessors import DataFramePreparer

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Telco Customer Churn – Model Comparison",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telco Customer Churn – Model Comparison Dashboard")
st.write("Comparación de modelos de ML e inferencia interactiva para churn.")

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

df = load_data()

TARGET = "Churn"
TARGET_MAP = {"No": 0, "Yes": 1}

# ======================================================
# FEATURES
# ======================================================
FEATURES_FULL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

FEATURES_TOP = [
    "TotalCharges", "tenure", "MonthlyCharges", "Contract",
    "TechSupport", "OnlineSecurity", "PaymentMethod",
    "PaperlessBilling", "gender", "InternetService", "Dependents"
]

# ======================================================
# CATEGORICAL OPTIONS (EXACT MATCH DATASET)
# ======================================================
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# ======================================================
# MODEL REGISTRY
# ======================================================
MODEL_REGISTRY = {
    "Logistic Regression": {
        "Completo": {"file": "logreg_full.pkl", "features": FEATURES_FULL},
        "Top features": {"file": "logreg_top.pkl", "features": FEATURES_TOP}
    },
    "Random Forest": {
        "Completo": {"file": "rf_full.pkl", "features": FEATURES_FULL},
        "Top features": {"file": "rf_top.pkl", "features": FEATURES_TOP}
    },
    "Gradient Boosting": {
        "Completo": {"file": "model3_full.pkl", "features": FEATURES_FULL},
        "Top features": {"file": "model3_top.pkl", "features": FEATURES_TOP}
    }
}

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("⚙️ Model Configuration")

model_name = st.sidebar.selectbox(
    "Select model",
    list(MODEL_REGISTRY.keys())
)

version_name = st.sidebar.radio(
    "Model version",
    ["Completo", "Top features"]
)

threshold = st.sidebar.slider(
    "Decision threshold",
    0.0, 1.0, 0.5, 0.01
)

cfg = MODEL_REGISTRY[model_name][version_name]
features_sel = cfg["features"]
model = load_model(cfg["file"])

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔮 Inferencia individual",
    "📊 Comparación de métricas",
    "🧩 Confusión & Importancia",
    "📈 EDA del dataset",
    "🏆 Comparativa Global de Modelos",
    "🧪 Simulador",
    "🎯 Decision Threshold"
])

# ======================================================
# 1️⃣ INDIVIDUAL PREDICTION
# ======================================================
with tab1:
    st.subheader("🔮 Individual Prediction")

    with st.form("prediction_form"):
        inputs = {}
        cols = st.columns(3)

        for i, col in enumerate(features_sel):
            with cols[i % 3]:
                if col == "SeniorCitizen":
                    inputs[col] = st.selectbox(col, [0, 1])
                elif col == "tenure":
                    inputs[col] = st.number_input(col, 0, 120, 12)
                elif col in ["MonthlyCharges", "TotalCharges"]:
                    inputs[col] = st.number_input(col, 0.0, 10000.0, 70.0)
                elif col in CATEGORICAL_OPTIONS:
                    inputs[col] = st.selectbox(col, CATEGORICAL_OPTIONS[col])

        submitted = st.form_submit_button("Predict")

    if submitted:
        # 🔥 CRÍTICO: crear TODAS las columnas esperadas por el pipeline
        row = {c: [inputs.get(c, None)] for c in FEATURES_FULL}
        df_new = pd.DataFrame(row)

        proba = model.predict_proba(df_new)[0, 1]
        pred = int(proba >= threshold)

        st.metric("Churn probability", f"{proba*100:.2f}%")
        st.metric("Prediction", "Churn = Yes" if pred else "Churn = No")

# ======================================================
# 2️⃣ FULL vs TOP METRICS
# ======================================================
with tab2:
    rows = []

    for vname, cfg_v in MODEL_REGISTRY[model_name].items():
        mdl = load_model(cfg_v["file"])
        feats = cfg_v["features"]

        df_eval = df.dropna(subset=[TARGET] + feats)
        X = df_eval[feats]
        y = df_eval[TARGET].map(TARGET_MAP)

        p = mdl.predict_proba(X)[:, 1]
        pred = (p >= threshold).astype(int)

        rows.append([
            vname,
            accuracy_score(y, pred),
            f1_score(y, pred),
            roc_auc_score(y, p)
        ])

    df_metrics = pd.DataFrame(rows, columns=["Version", "Accuracy", "F1", "AUC"])
    st.dataframe(df_metrics.round(3))

    fig = px.bar(
        df_metrics.melt(id_vars="Version"),
        x="variable", y="value",
        color="Version",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 3️⃣ CONFUSION + FEATURE IMPORTANCE
# ======================================================
with tab3:
    X = df[features_sel].dropna()
    y = df.loc[X.index, TARGET].map(TARGET_MAP)

    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)

    cm = confusion_matrix(y, pred)
    st.write("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm))

    clf = model.named_steps.get("clf", model)
    prep = model.named_steps.get("prep", None)

    if hasattr(clf, "feature_importances_"):
        names = prep._columns if prep else features_sel
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        names = prep._columns if prep else features_sel
        imp = np.abs(clf.coef_[0])
    else:
        imp = None

    if imp is not None:
        df_imp = pd.DataFrame({
            "Feature": names,
            "Importance": imp
        }).sort_values("Importance", ascending=False)

        st.dataframe(df_imp.head(15))
        fig = px.bar(df_imp.head(15), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 4️⃣ EDA
# ======================================================
with tab4:
    st.subheader("📊 Dashboard General – Visión global del dataset")

    # -----------------------------
    # 1) Preparación base
    # -----------------------------
    df_dash = df.copy()

    # Normalizar TotalCharges si existiera
    if "TotalCharges" in df_dash.columns:
        df_dash["TotalCharges"] = pd.to_numeric(df_dash["TotalCharges"], errors="coerce")

    # Binario churn
    if TARGET not in df_dash.columns:
        st.error(f"No existe la columna objetivo '{TARGET}' en el dataset.")
        st.stop()

    df_dash["_churn_bin"] = df_dash[TARGET].map({"No": 0, "Yes": 1})

    # Limpiar nulos críticos
    df_dash = df_dash.dropna(subset=["_churn_bin"]).copy()

    # -----------------------------
    # 2) KPIs principales
    # -----------------------------
    total_clientes = len(df_dash)
    churn_total = int(df_dash["_churn_bin"].sum())
    churn_rate = float(df_dash["_churn_bin"].mean()) if total_clientes else 0.0

    avg_monthly = float(df_dash["MonthlyCharges"].dropna().mean()) if "MonthlyCharges" in df_dash.columns else np.nan
    avg_total = float(df_dash["TotalCharges"].dropna().mean()) if "TotalCharges" in df_dash.columns else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total clientes", f"{total_clientes:,}")
    c2.metric("Churn (Yes)", f"{churn_total:,}")
    c3.metric("Churn rate", f"{churn_rate*100:.2f}%")
    c4.metric("Avg MonthlyCharges", "-" if np.isnan(avg_monthly) else f"{avg_monthly:.2f}")
    c5.metric("Avg TotalCharges", "-" if np.isnan(avg_total) else f"{avg_total:.2f}")

    st.markdown("---")

    # -----------------------------
    # 3) Distribución de churn
    # -----------------------------
    st.write("### Distribución de la variable objetivo (Churn)")
    fig_churn = px.histogram(df_dash, x=TARGET, color=TARGET, barmode="group")
    st.plotly_chart(fig_churn, use_container_width=True)

    # -----------------------------
    # 4) Comparación numéricas por churn
    # -----------------------------
    st.write("### Variables numéricas vs Churn")
    num_candidates = []
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df_dash.columns:
            num_candidates.append(col)

    if len(num_candidates) > 0:
        col_num = st.selectbox("Selecciona variable numérica", num_candidates, index=0, key="tab8_num")
        fig_box = px.box(df_dash.dropna(subset=[col_num]), x=TARGET, y=col_num, points="outliers")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No se encontraron variables numéricas esperadas (tenure/MonthlyCharges/TotalCharges).")

    st.markdown("---")

    # -----------------------------
    # 5) Churn rate por una dimensión (barra)
    # -----------------------------
    st.write("### Churn rate por segmento")
    cat_options = [
        "Contract", "InternetService", "PaymentMethod", "TechSupport",
        "OnlineSecurity", "Partner", "Dependents", "gender", "SeniorCitizen"
    ]
    cat_options = [c for c in cat_options if c in df_dash.columns]

    if len(cat_options) == 0:
        st.info("No hay variables categóricas esperadas para segmentar.")
    else:
        dim = st.selectbox("Segmentar por", cat_options, index=0, key="tab8_dim1")
        seg = (
            df_dash.dropna(subset=[dim])
            .groupby(dim)["_churn_bin"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "N"})
            .sort_values("ChurnRate", ascending=False)
        )
        fig_seg = px.bar(seg, x=dim, y="ChurnRate", text=seg["ChurnRate"].round(3))
        st.plotly_chart(fig_seg, use_container_width=True)
        st.dataframe(seg, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 6) Heatmap: churn rate por dos dimensiones (pivot)
    # -----------------------------
    st.write("### Heatmap: Churn rate por 2 dimensiones (segmentación cruzada)")

    if len(cat_options) < 2:
        st.info("Necesitas al menos 2 variables categóricas para crear el heatmap.")
        st.stop()

    dim_a = st.selectbox("Dimensión A (filas)", cat_options, index=0, key="tab8_dim_a")
    dim_b = st.selectbox("Dimensión B (columnas)", cat_options, index=1, key="tab8_dim_b")

    # ✅ Evita el error: Grouper not 1-dimensional
    if dim_a == dim_b:
        st.warning("La Dimensión A y la Dimensión B no pueden ser la misma. Selecciona dos diferentes.")
        st.stop()

    df_piv = df_dash.dropna(subset=[dim_a, dim_b, "_churn_bin"]).copy()

    pv = pd.pivot_table(
        df_piv,
        index=dim_a,
        columns=dim_b,
        values="_churn_bin",
        aggfunc="mean"
    )

    fig_hm = px.imshow(
        pv,
        aspect="auto",
        text_auto=".2f",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.write("Tabla del heatmap (churn rate):")
    st.dataframe(pv.round(3), use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 7) Top segmentos “más riesgosos” (tabla)
    # -----------------------------
    st.write("### Top segmentos con mayor churn (A x B)")

    tmp = (
        df_piv
        .groupby([dim_a, dim_b])["_churn_bin"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "ChurnRate", "count": "N"})
        .sort_values(["ChurnRate", "N"], ascending=[False, False])
    )

    top_n = st.slider("Top N segmentos", 5, 30, 10, key="tab8_topn")
    st.dataframe(tmp.head(top_n).round({"ChurnRate": 3}), use_container_width=True)


# ======================================================
# 5️⃣ GLOBAL COMPARISON
# ======================================================
# ======================================================
# 5️⃣ GLOBAL COMPARISON (CRITERIO: F1-SCORE)
# ======================================================
with tab5:
    st.subheader("🏆 Global Model Comparison (F1-Score )")

    rows = []

    for mdl_name, versions in MODEL_REGISTRY.items():
        for vname, cfg_v in versions.items():
            try:
                mdl = load_model(cfg_v["file"])
            except Exception as e:
                st.warning(f"No se pudo cargar {cfg_v['file']}: {e}")
                continue

            feats = cfg_v["features"]
            df_eval = df.dropna(subset=[TARGET] + feats)

            X = df_eval[feats]
            y = df_eval[TARGET].map(TARGET_MAP)

            try:
                proba = mdl.predict_proba(X)[:, 1]
            except Exception:
                continue

            pred = (proba >= 0.5).astype(int)

            rows.append([
                mdl_name,
                vname,
                accuracy_score(y, pred),
                f1_score(y, pred),
                roc_auc_score(y, proba)
            ])

    df_all = pd.DataFrame(
        rows,
        columns=["Model", "Version", "Accuracy", "F1", "AUC"]
    )

    # ============================
    # 📋 TABLA GLOBAL
    # ============================
    st.write("### 📋 Global comparison table")

    num_cols = df_all.select_dtypes(include=np.number).columns
    st.dataframe(df_all.style.format({c: "{:.3f}" for c in num_cols}))

    # ============================
    # 🏆 RANKING POR F1-SCORE
    # ============================
    st.write("### 🏆 Model ranking based on F1-Score")

    df_f1 = df_all.sort_values("F1", ascending=False)

    fig_f1 = px.bar(
        df_f1,
        x="F1",
        y="Model",
        color="Version",
        orientation="h",
        text=df_f1["F1"].round(3),
        title="Global Model Ranking – F1-Score (Higher is Better)"
    )

    fig_f1.update_layout(
        xaxis_title="F1-Score",
        yaxis_title="Model",
        legend_title="Version",
        height=500,
        xaxis_range=[0, 1]
    )

    st.plotly_chart(fig_f1, use_container_width=True)

    # ============================
    # 📊 MÉTRICAS DE APOYO
    # ============================
    st.write("### 📊 Supporting metrics (Accuracy & AUC)")

    df_support = df_all.melt(
        id_vars=["Model", "Version"],
        value_vars=["Accuracy", "AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig_support = px.bar(
        df_support,
        x="Metric",
        y="Score",
        color="Version",
        barmode="group",
        facet_col="Model",
        text=df_support["Score"].round(3),
        title="Supporting Metrics per Model"
    )

    fig_support.update_layout(
        height=550,
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig_support, use_container_width=True)

    # ============================
    # 🧠 CONCLUSIÓN AUTOMÁTICA
    # ============================
    best_model = df_f1.iloc[0]

    st.success(
        f"🏆 **Best overall model (based on F1-Score):** "
        f"{best_model['Model']} – {best_model['Version']} "
        f"(F1 = {best_model['F1']:.3f})"
    )

# ======================================================
# 6️⃣ WHAT-IF SIMULATOR
# ======================================================
with tab6:
    st.subheader("🧪 Simulador – Ajusta variables y observa el impacto")

    if not model :
        st.warning("No se pudo cargar el modelo seleccionado.")
    else:
        st.write("Modifica las características y observa cómo cambia la probabilidad de churn.")

        # Variables clave que más afectan el churn
        c1, c2, c3 = st.columns(3)

        with c1:
            tenure_wi = st.slider("Meses en la compañía (tenure)", 0, 72, 12)
            contract_wi = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        with c2:
            monthly_wi = st.slider("Cargo mensual", 0, 120, 70)
            tech_wi = st.selectbox("Soporte técnico", ["No", "Yes"])
        with c3:
            internet_wi = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
            security_wi = st.selectbox("OnlineSecurity", ["No", "Yes"])

        # Construimos fila
        sim_dict = {
            col: [None] for col in features_sel  # default
        }
        def setv(col, val):
            if col in sim_dict:
                sim_dict[col][0] = val

        # Asignar solo columnas relevantes
        setv("tenure", tenure_wi)
        setv("Contract", contract_wi)
        setv("MonthlyCharges", monthly_wi)
        setv("TechSupport", tech_wi)
        setv("OnlineSecurity", security_wi)
        setv("InternetService", internet_wi)

        df_sim = pd.DataFrame(sim_dict)

        try:
            proba_sim = model.predict_proba(df_sim)[0, 1]

            st.metric("Probabilidad estimada de churn", f"{proba_sim*100:.2f} %")
        except Exception as e:
            st.error(f"Error al calcular la probabilidad: {e}")
            proba_sim = None

        # ====================================================
        # Reglas de recomendación inteligentes
        # ====================================================
        st.write("### 💡 Recomendaciones de retención")

        recs = []

        if tenure_wi < 12:
            recs.append("Cliente nuevo: ofrecer beneficios de bienvenida o descuentos por 3 meses.")

        if contract_wi == "Month-to-month":
            recs.append("Sugerir migración a contrato anual para reducir riesgo de churn.")

        if monthly_wi > 85:
            recs.append("Cliente con cargo alto: ofrecer revisión del plan o upgrade a fibra con descuento.")

        if tech_wi == "No":
            recs.append("Ofrecer soporte técnico básico gratuito por 1 mes.")

        if security_wi == "No":
            recs.append("Incentivar paquetes de seguridad online para aumentar permanencia.")

        if internet_wi == "DSL":
            recs.append("Ofrecer cambio a fibra óptica para mejorar experiencia y reducir churn.")

        if len(recs) == 0:
            st.success("El cliente no muestra señales significativas de riesgo. Mantener contacto regular.")
        else:
            for r in recs:
                st.info("• " + r)
from sklearn.metrics import roc_curve, precision_recall_curve
with tab7:
    st.subheader("🎚️ Decision Threshold – Curvas y Trade-off")

    # 1) Validar modelo
    if model is None:
        st.warning("No se pudo cargar el modelo seleccionado.")
        st.stop()

    # 2) Dataset de evaluación consistente con features_sel
    df_eval = df.dropna(subset=[TARGET] + features_sel).copy()
    X_eval = df_eval[features_sel]
    y_eval = df_eval[TARGET].map(TARGET_MAP).astype(int).values

    # 3) Probabilidades
    try:
        y_proba = model.predict_proba(X_eval)[:, 1]
    except Exception as e:
        st.error(f"El modelo no pudo calcular predict_proba: {e}")
        st.stop()

    st.caption(f"Evaluación: {len(y_eval)} filas, {len(features_sel)} features.")

    st.markdown("""
    Los modelos producen **probabilidades**, no clases.
    El *decision threshold* define desde qué probabilidad se clasifica como **Churn = Yes**.

    Ajustarlo controla el trade-off:
    - **Precision** (menos falsos positivos)
    - **Recall** (menos falsos negativos)
    """)

    # ==========================
    # 1) ROC Curve
    # ==========================
    fpr, tpr, _ = roc_curve(y_eval, y_proba)
    auc_score = roc_auc_score(y_eval, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_score:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # ==========================
    # 2) Precision–Recall Curve
    # ==========================
    pr_precision, pr_recall, _ = precision_recall_curve(y_eval, y_proba)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name="PR Curve"))
    fig_pr.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True)

    # ==========================
    # 3) F1 vs Threshold
    # ==========================
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []

    for thr in thresholds:
        pred_thr = (y_proba >= thr).astype(int)
        f1_scores.append(f1_score(y_eval, pred_thr))

    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode="lines", name="F1"))

    fig_f1.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Selected threshold = {threshold:.2f}"
    )

    fig_f1.update_layout(title="F1 Score vs Decision Threshold", xaxis_title="Threshold", yaxis_title="F1 Score")
    st.plotly_chart(fig_f1, use_container_width=True)

    # ==========================
    # Métricas en threshold actual
    # ==========================
    pred_selected = (y_proba >= threshold).astype(int)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{precision_score(y_eval, pred_selected):.3f}")
    col2.metric("Recall", f"{recall_score(y_eval, pred_selected):.3f}")
    col3.metric("F1", f"{f1_score(y_eval, pred_selected):.3f}")
    col4.metric("AUC", f"{roc_auc_score(y_eval, y_proba):.3f}")

    st.markdown("---")
    st.subheader("📋 Tabla FP vs Detectados por Threshold (para minimizar FP)")

    thr_grid = np.linspace(0.05, 0.95, 30)
    rows = []

    for thr in thr_grid:
        pred_thr = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_eval, pred_thr).ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1v = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        rows.append({
            "Threshold": round(float(thr), 2),
            "TP (Detectados churn)": int(tp),
            "FP (Falsos positivos)": int(fp),
            "FN (Churn perdidos)": int(fn),
            "Precision": round(float(prec), 3),
            "Recall": round(float(rec), 3),
            "F1": round(float(f1v), 3),
        })

    df_thr = pd.DataFrame(rows)

    st.dataframe(df_thr, use_container_width=True)

    st.write("### Top thresholds con menos FP (desempate por mayor F1)")
    df_less_fp = df_thr.sort_values(["FP (Falsos positivos)", "F1"], ascending=[True, False]).head(10)
    st.dataframe(df_less_fp, use_container_width=True)

    best = df_less_fp.iloc[0]
    st.info(
        f"Sugerencia: Threshold={best['Threshold']:.2f} "
        f"→ FP={best['FP (Falsos positivos)']}, TP={best['TP (Detectados churn)']}, F1={best['F1']:.3f}"
    )

    fig_fp = px.line(df_thr, x="Threshold", y="FP (Falsos positivos)", title="Falsos Positivos vs Threshold")
    st.plotly_chart(fig_fp, use_container_width=True)

# ==========================================================
# 8) DASHBOARD GENERAL DEL DATASET (TAB 8)
# ==========================================================

    