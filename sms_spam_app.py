"""
sms_spam_app.py
Single-file project: dataset download -> train -> save -> Streamlit UI

Run:
    streamlit run sms_spam_app.py

Behavior:
- On first run: downloads dataset, trains models, saves best model + vectorizer.
- Streamlit UI allows single message classification (and batch via textarea).
- Option to retrain models from the UI.
"""

import os
import io
import zipfile
import tempfile
import requests
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

import re
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import streamlit as st

# Try to import nltk tools; download if not present
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------- Config / File paths ----------
MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
DATA_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE_NAME = "SMSSpamCollection"  # inside zip

# --------- Utility functions ----------
def download_and_load_dataset(url: str = DATA_ZIP_URL) -> pd.DataFrame:
    """
    Download the UCI SMS Spam Collection dataset (zip) and return a DataFrame with columns: label, message
    """
    st.info("Downloading dataset..." if st._is_running_with_streamlit else "Downloading dataset...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(DATA_FILE_NAME) as f:
        df = pd.read_csv(f, sep="\t", header=None, names=["label", "message"], quoting=3)
    return df

def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Basic text cleaning: lowercasing, removing URLs, non-alphanumeric chars, stopwords removal (optional lemmatization)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove URLs and emails
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # remove non-alphanumeric (keeps spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def prepare_data(df: pd.DataFrame, do_lemmatize: bool = True) -> pd.DataFrame:
    df = df.copy()
    st.info("Preprocessing data..." if st._is_running_with_streamlit else "Preprocessing data...")
    # ensure label is binary 1=spam, 0=ham
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    # Clean messages
    # apply in batches for speed if large; here dataset is small
    df["clean_msg"] = df["message"].apply(lambda x: clean_text(x, lemmatize=do_lemmatize))
    return df

def train_and_select_model(X_train, y_train, X_test, y_test, max_features=5000) -> Tuple[Pipeline, dict]:
    """
    Train two pipelines: MultinomialNB and LogisticRegression with TF-IDF.
    Returns (best_pipeline, metrics_dict)
    """
    st.info("Training models..." if st._is_running_with_streamlit else "Training models...")
    # Pipelines
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))

    pipe_nb = Pipeline([
        ("tfidf", tfidf),
        ("clf", MultinomialNB())
    ])

    pipe_lr = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    # Train
    pipe_nb.fit(X_train, y_train)
    pipe_lr.fit(X_train, y_train)

    # Evaluate
    preds_nb = pipe_nb.predict(X_test)
    preds_lr = pipe_lr.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

    m_nb = metrics(y_test, preds_nb)
    m_lr = metrics(y_test, preds_lr)

    # choose best by f1
    best_pipe = pipe_lr if m_lr["f1"] >= m_nb["f1"] else pipe_nb
    best_metrics = m_lr if best_pipe is pipe_lr else m_nb

    # prepare report strings
    report_nb = classification_report(y_test, preds_nb, target_names=["ham","spam"])
    report_lr = classification_report(y_test, preds_lr, target_names=["ham","spam"])

    results = {
        "nb": {"metrics": m_nb, "report": report_nb},
        "lr": {"metrics": m_lr, "report": report_lr},
        "best": {"pipeline": best_pipe, "metrics": best_metrics, "which": "LogisticRegression" if best_pipe is pipe_lr else "MultinomialNB"}
    }

    return best_pipe, results

# --------- Main training flow ----------
def ensure_nltk_resources():
    try:
        _ = stopwords.words("english")
        _ = WordNetLemmatizer()
    except LookupError:
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

def train_pipeline(save_model_path: str = MODEL_FILE, save_vect_path: str = VECTORIZER_FILE, retrain: bool = False):
    ensure_nltk_resources()

    # If model exists and not retrain, skip training
    if os.path.exists(save_model_path) and os.path.exists(save_vect_path) and not retrain:
        st.success("Model and vectorizer already exist. Skipping training.")
        model = joblib.load(save_model_path)
        vect = joblib.load(save_vect_path)
        return model, vect, None

    # Download dataset
    df = download_and_load_dataset()

    # Basic EDA: shape, imbalance info
    if st._is_running_with_streamlit:
        st.write("Dataset snapshot:")
        st.write(df.head())
        st.write("Class distribution:")
        st.write(df["label"].value_counts())
    else:
        print("Dataset shape:", df.shape)
        print(df["label"].value_counts())

    # Preprocess
    df = prepare_data(df, do_lemmatize=True)

    # Train-test split
    X = df["clean_msg"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train and pick best model
    best_pipeline, results = train_and_select_model(X_train, y_train, X_test, y_test, max_features=5000)

    # Save model & vectorizer
    joblib.dump(best_pipeline, save_model_path)
    # Extract TF-IDF from pipeline and save separately
    try:
        tfidf = best_pipeline.named_steps["tfidf"]
        joblib.dump(tfidf, save_vect_path)
    except Exception:
        # if pipeline structure different, just save pipeline as vectorizer fallback
        joblib.dump(best_pipeline, save_vect_path)

    # Return model, vectorizer, results
    return best_pipeline, tfidf, results

# --------- Streamlit UI ----------
def run_streamlit_app():
    st.set_page_config(page_title="SMS Spam Detector", layout="centered")

    st.title("📨 SMS Spam Detection — Eduvate Oracle Internship")
    st.markdown(
        """
    This demo downloads the **SMS Spam Collection** dataset, trains a model (Naive Bayes + Logistic Regression),
    selects the best model, and exposes a Streamlit UI for inference.
    """
    )

    # Sidebar controls
    st.sidebar.header("Controls")
    retrain = st.sidebar.button("Retrain model (download + train)")
    show_metrics = st.sidebar.checkbox("Show training evaluation metrics", value=True)

    # Train or load model
    if retrain or (not (os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE))):
        with st.spinner("Training (this may take ~30–90 seconds depending on your machine)..."):
            model, vect, results = train_pipeline(retrain=True)
    else:
        # load saved
        model = joblib.load(MODEL_FILE)
        vect = joblib.load(VECTORIZER_FILE)
        results = None

    if results is None:
        # if we didn't just train, try to load a stored results summary if available
        st.info("Model loaded from disk.")
    else:
        if show_metrics:
            st.subheader("Model evaluation summary")
            st.write("Best model:", results["best"]["which"])
            st.write("Best model metrics (on test set):")
            st.json(results["best"]["metrics"])

            st.write("---")
            st.write("MultinomialNB classification report:")
            st.text(results["nb"]["report"])
            st.write("LogisticRegression classification report:")
            st.text(results["lr"]["report"])

    st.write("---")
    st.subheader("Classify a single SMS message")
    input_sms = st.text_area("Enter SMS message here", height=120, placeholder="e.g. Free entry in 2 a weekly competition to win FA Cup...")
    if st.button("Classify"):
        if not input_sms or len(input_sms.strip()) == 0:
            st.warning("Please enter some text to classify.")
        else:
            # preprocess same as training
            ensure_nltk_resources()
            cleaned = clean_text(input_sms)
            # vectorize using pipeline (model already contains tfidf), so call predict_proba via pipeline
            try:
                proba = model.predict_proba([cleaned])[0]
                pred = model.predict([cleaned])[0]
                label = "spam" if pred == 1 else "ham"
                confidence = float(proba[pred])
            except Exception:
                # fallback: if pipeline predict_proba not available, use decision_function or set confidence to 1.0
                pred = model.predict([cleaned])[0]
                label = "spam" if pred == 1 else "ham"
                confidence = 1.0

            st.markdown(f"**Prediction:** `{label.upper()}`")
            st.markdown(f"**Confidence:** {confidence:.3f}")
            st.write("**Cleaned input used for prediction:**")
            st.write(cleaned)

    st.write("---")
    st.subheader("Batch classify (paste multiple messages, one per line)")
    batch_input = st.text_area("Enter multiple messages (one per line)", height=160)
    if st.button("Classify Batch"):
        lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            ensure_nltk_resources()
            cleaned_lines = [clean_text(l) for l in lines]
            preds = model.predict(cleaned_lines)
            try:
                probas = model.predict_proba(cleaned_lines)
            except Exception:
                probas = None
            df_out = pd.DataFrame({
                "message": lines,
                "cleaned": cleaned_lines,
                "pred_label": ["spam" if p==1 else "ham" for p in preds],
                "confidence": [float(probas[i][preds[i]]) if probas is not None else None for i in range(len(preds))]
            })
            st.dataframe(df_out)

    st.write("---")
    st.caption("Model and vectorizer saved as `spam_model.pkl` and `tfidf_vectorizer.pkl` in the current directory.")

# --------- CLI fallback for training outside Streamlit ----------
def cli_train():
    print("Running training (CLI mode)...")
    ensure_nltk_resources()
    model, vect, results = train_pipeline(retrain=False)
    print("Training finished. Best model:", results["best"]["which"])
    print("Best metrics:", results["best"]["metrics"])
    print("Models saved as:", MODEL_FILE, VECTORIZER_FILE)

# --------- Entrypoint ----------
if __name__ == "__main__":
    # If run via streamlit, Streamlit will rerun and set environment; check presence of STREAMLIT_SERVER_PORT
    if "STREAMLIT_RUN" in os.environ or ("streamlit" in os.path.basename(__file__).lower()) or ("STREAMLIT_SERVER_RUNNING" in os.environ):
        # run streamlit app (this branch may be active when streamlit runs)
        run_streamlit_app()
    else:
        # If executed directly (python sms_spam_app.py) show instructions
        print("This script is intended to be run with Streamlit.")
        print("Run with: streamlit run sms_spam_app.py")
        # offer to train from CLI if user wants
        ans = input("Type 'train' to train the model now from CLI, or press Enter to exit: ").strip().lower()
        if ans == "train":
            cli_train()
        else:
            print("Exiting. To run the app: streamlit run sms_spam_app.py")
