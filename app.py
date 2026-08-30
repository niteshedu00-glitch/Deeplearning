import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris ANN Classifier",
    page_icon="🌸",
    layout="wide",
)

SPECIES_INFO = {
    "Iris-setosa": {"emoji": "🌷", "color": "#7FB3D5"},
    "Iris-versicolor": {"emoji": "🌸", "color": "#82E0AA"},
    "Iris-virginica": {"emoji": "🌺", "color": "#F1948A"},
}


# --------------------------------------------------------------------------
# Data & model (cached so it only trains once per session)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    return df


@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Species", "Id"])
    y = df["Species"]

    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform, not fit_transform, on test

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    tf.random.set_seed(42)
    model = Sequential([
        Dense(16, input_dim=4, activation="relu"),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train_scaled, y_train_cat,
        epochs=100, batch_size=8, validation_split=0.2, verbose=0,
    )

    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "history": history,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "report": report,
        "cm": cm,
        "X": X,
    }


df = load_data()
artifacts = train_model(df)
model = artifacts["model"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🌸 Iris Species Classifier — Artificial Neural Network")
st.caption(
    "A Keras Sequential ANN (16 → 8 → 3, ReLU + Softmax) trained on the classic Iris dataset."
)

tab_predict, tab_explore, tab_performance = st.tabs(
    ["🔮 Predict", "📊 Explore Data", "📈 Model Performance"]
)

# --------------------------------------------------------------------------
# TAB 1: Predict
# --------------------------------------------------------------------------
with tab_predict:
    col_inputs, col_result = st.columns([1, 1.2], gap="large")

    with col_inputs:
        st.subheader("Flower Measurements")
        st.write("Adjust the sliders to describe a flower's measurements (in cm).")

        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

        st.divider()
        st.caption("Or try a preset example:")
        preset_cols = st.columns(3)
        presets = {
            "Setosa-like": (5.0, 3.4, 1.5, 0.2),
            "Versicolor-like": (6.0, 2.8, 4.3, 1.3),
            "Virginica-like": (6.7, 3.0, 5.5, 2.1),
        }
        for c, (label, vals) in zip(preset_cols, presets.items()):
            if c.button(label, use_container_width=True):
                sepal_length, sepal_width, petal_length, petal_width = vals
                st.session_state["_preset"] = vals

    with col_result:
        st.subheader("Prediction")

        input_arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_arr)
        probs = model.predict(input_scaled, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_species = encoder.classes_[pred_idx]
        confidence = probs[pred_idx] * 100

        info = SPECIES_INFO.get(pred_species, {"emoji": "🌼", "color": "#AAB7B8"})

        st.markdown(
            f"""
            <div style="padding:24px;border-radius:14px;background-color:{info['color']}22;
                        border:2px solid {info['color']};text-align:center;">
                <div style="font-size:52px;">{info['emoji']}</div>
                <div style="font-size:26px;font-weight:700;color:#222;">{pred_species}</div>
                <div style="font-size:16px;color:#555;">Confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.write("**Class probabilities**")
        prob_df = pd.DataFrame({
            "Species": encoder.classes_,
            "Probability": probs,
        }).sort_values("Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(5, 2.2))
        colors = [SPECIES_INFO.get(s, {"color": "#AAB7B8"})["color"] for s in prob_df["Species"]]
        ax.barh(prob_df["Species"], prob_df["Probability"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
        st.pyplot(fig, use_container_width=True)

        with st.expander("Show input as a table"):
            st.dataframe(
                pd.DataFrame(input_arr, columns=[
                    "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"
                ]),
                use_container_width=True, hide_index=True,
            )

# --------------------------------------------------------------------------
# TAB 2: Explore Data
# --------------------------------------------------------------------------
with tab_explore:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(df))
    c2.metric("Features", 4)
    c3.metric("Classes", df["Species"].nunique())
    c4.metric("Missing Values", int(df.isnull().sum().sum()))

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.write("**Class distribution**")
        st.bar_chart(df["Species"].value_counts())

        st.write("**Sample rows**")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    with right:
        st.write("**Feature relationships**")
        feature_x = st.selectbox(
            "X-axis", ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], index=2
        )
        feature_y = st.selectbox(
            "Y-axis", ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], index=3
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(
            data=df, x=feature_x, y=feature_y, hue="Species",
            palette=[SPECIES_INFO[s]["color"] for s in df["Species"].unique()] if set(df["Species"].unique()) <= set(SPECIES_INFO) else None,
            ax=ax2, s=60,
        )
        st.pyplot(fig2, use_container_width=True)

    with st.expander("Full pairplot (all features)"):
        with st.spinner("Rendering pairplot..."):
            pair_fig = sns.pairplot(df.drop(columns=["Id"]), hue="Species")
            st.pyplot(pair_fig)

# --------------------------------------------------------------------------
# TAB 3: Model Performance
# --------------------------------------------------------------------------
with tab_performance:
    st.subheader("Training & Evaluation")

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy", f"{artifacts['test_acc']*100:.2f}%")
    m2.metric("Test Loss", f"{artifacts['test_loss']:.4f}")
    m3.metric("Epochs Trained", len(artifacts["history"].history["accuracy"]))

    left, right = st.columns(2, gap="large")

    with left:
        st.write("**Accuracy over epochs**")
        hist = artifacts["history"].history
        hist_df = pd.DataFrame({
            "train_accuracy": hist["accuracy"],
            "val_accuracy": hist["val_accuracy"],
        })
        st.line_chart(hist_df)

        st.write("**Loss over epochs**")
        loss_df = pd.DataFrame({
            "train_loss": hist["loss"],
            "val_loss": hist["val_loss"],
        })
        st.line_chart(loss_df)

    with right:
        st.write("**Confusion Matrix (test set)**")
        fig3, ax3 = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(
            artifacts["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax3,
        )
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        st.pyplot(fig3, use_container_width=True)

    st.write("**Classification Report**")
    report_df = pd.DataFrame(artifacts["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

st.divider()
st.caption("Built from the notebook's Keras Sequential ANN (Dense 16 → 8 → 3, Adam optimizer, categorical crossentropy).")