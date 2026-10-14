"""Inspect held-out results and predict survival from a local CSV."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import streamlit as st

from model import load_bundle, predict, score

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


@st.cache_resource
def saved_model(signature):
    """Reload the model when its saved files change."""
    return load_bundle(ARTIFACTS)


def show_evaluation(labels, probabilities, metrics):
    """Show metrics and plots for the same labeled passenger rows."""
    for column, key in zip(st.columns(4), ("accuracy", "precision", "recall", "f1")):
        column.metric(key.title(), f"{metrics[key]:.1%}")
    auc = metrics["roc_auc"]
    interval = metrics.get("accuracy_ci95")
    details = f"{len(labels)} passengers · ROC AUC: {auc:.3f}" if auc is not None else (
        f"{len(labels)} passengers · ROC AUC unavailable: only one outcome is present."
    )
    details += f" · Brier score: {metrics['brier']:.3f} (lower is better)"
    if interval:
        details += f" · Accuracy 95% interval: {interval[0]:.1%}–{interval[1]:.1%}"
    st.caption(details)

    confusion_column, roc_column = st.columns(2)
    with confusion_column:
        matrix = np.asarray(metrics["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(4.8, 3.8))
        axis.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                          color="white" if matrix[row, column] > matrix.max() / 2 else "black")
        axis.set(xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["Did not survive", "Survived"],
                 yticklabels=["Did not survive", "Survived"],
                 xlabel="Predicted outcome", ylabel="Actual outcome", title="Confusion matrix")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with roc_column:
        if auc is None:
            st.info("A ROC curve needs both survival outcomes.")
        else:
            false_positive, true_positive, _ = roc_curve(labels, probabilities)
            figure, axis = plt.subplots(figsize=(4.8, 3.8))
            axis.plot(false_positive, true_positive, color="#24628C", label=f"ROC AUC = {auc:.3f}")
            axis.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
            axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
                     ylabel="True positive rate", title="ROC curve")
            axis.legend(loc="lower right")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)


def validation_report(metrics):
    validation = pd.read_csv(ARTIFACTS / "validation_predictions.csv")
    st.subheader("Validation results")
    st.caption("Final evaluation on passengers excluded from model selection and training. "
               "Survived = 1; probability threshold = 0.5.")
    show_evaluation(validation["Survived"], validation["probability"], metrics["holdout"])
    if "majority_holdout" in metrics:
        st.caption(f"Majority-class reference accuracy: {metrics['majority_holdout']['accuracy']:.1%}. "
                   "The interval assumes independent passengers; shared travel groups reduce independence.")

    grouped = metrics.get("grouped_cv")
    if grouped:
        st.subheader("New travel groups")
        st.write(f"Development cross-validation accuracy: **{grouped['accuracy']:.1%}**; "
                 f"F1: **{grouped['f1']:.1%}**.")
        st.caption("Related passengers can share tickets or families. Grouped cross-validation "
                   "keeps connected travel groups together; the random validation split may share "
                   "groups with training. These evaluations answer different generalization questions.")

    comparison = metrics.get("development_cv")
    if comparison:
        with st.expander("Compare models"):
            st.caption("These scores guided selection, so the winning development score can be optimistic. "
                       "The final validation set was reserved until selection was complete.")
            columns = ["name", "kind", "level", "accuracy", "f1", "roc_auc", "fold_accuracy_std"]
            table = pd.DataFrame(comparison).reindex(columns=columns)
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={key: st.column_config.NumberColumn(format="%.3f")
                                        for key in columns[3:]})
    st.download_button("Download validation predictions", validation.to_csv(index=False),
                       "validation_predictions.csv", "text/csv")


def inference(bundle):
    st.subheader("CSV input")
    st.write("Enter a CSV path on the machine running this app. Include the original Titanic "
             "predictor columns; `Survived` and `PassengerId` are optional.")
    with st.form("inference"):
        source = st.text_input("CSV path", value="artifacts/validation_predictions.csv")
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        st.session_state.pop("inference_result", None)
        try:
            path = Path(source).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            frame = pd.read_csv(path)
            if frame.empty:
                raise ValueError("The CSV has no passenger rows.")
            if "Survived" in frame and not frame["Survived"].isin([0, 1]).all():
                raise ValueError("Survived must contain only 0 or 1, with no missing labels.")
            probabilities = predict(frame, bundle)
            result = frame.assign(probability=probabilities,
                                  prediction=(probabilities >= 0.5).astype(int))
            st.session_state["inference_result"] = (str(path), result)
        except FileNotFoundError:
            st.error("CSV not found. Check the path and try again.")
        except (ValueError, OSError) as error:
            st.error(f"Unable to process this CSV: {error}")

    if "inference_result" not in st.session_state:
        return
    source_path, result = st.session_state["inference_result"]
    st.caption(f"Source: {source_path} · {len(result)} passengers · probability threshold = 0.5")
    development_ids = set(bundle[2].get("development_ids", []))
    if "PassengerId" in result:
        overlap = result["PassengerId"].isin(development_ids).sum()
        if overlap:
            st.warning(f"{overlap} passengers appeared in training. Metrics on these rows "
                       "do not measure performance on unseen passengers.")
    elif "Survived" in result:
        st.info("PassengerId is absent, so overlap with training passengers cannot be checked.")
    if "Survived" in result:
        show_evaluation(result["Survived"], result["probability"],
                        score(result["Survived"], result["probability"]))
    else:
        st.info("Predictions are ready. Add a binary Survived column to view evaluation metrics.")
    st.dataframe(result, hide_index=True, width="stretch")
    st.download_button("Download predictions", result.to_csv(index=False),
                       "predictions.csv", "text/csv")


def main():
    st.set_page_config(page_title="Titanic survival", layout="wide")
    st.markdown("""<style>
    .block-container {max-width: 1080px; padding-top: 2.2rem;}
    h1 {font-size: 2rem !important; letter-spacing: -0.02em;}
    h3 {font-size: 1.15rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>""", unsafe_allow_html=True)
    st.title("Titanic survival")
    st.caption("PyTorch classifier · 712 development passengers · 179 reserved for validation")
    try:
        signature = tuple((ARTIFACTS / name).stat().st_mtime_ns for name in (
            "model.pt", "preprocessor.joblib", "manifest.json"
        ))
        bundle = saved_model(signature)
        metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Model artifacts are unavailable: {error}")
        st.info("Run python train.py in the project folder, then reload this page.")
        st.stop()
    validation_tab, inference_tab = st.tabs(["Validation", "Predict from CSV"])
    with validation_tab:
        try:
            validation_report(metrics)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Validation results are unavailable: {error}")
    with inference_tab:
        inference(bundle)


if __name__ == "__main__":
    main()
