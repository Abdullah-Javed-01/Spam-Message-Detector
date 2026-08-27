from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "spam.csv"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42

# =========================================================
# 2. LOAD AND INSPECT DATASET
# =========================================================

df = pd.read_csv(DATASET_PATH, encoding="latin-1")

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nColumns found:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

# =========================================================
# 3. KEEP AND RENAME USEFUL COLUMNS
# =========================================================

if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]

elif "label" in df.columns and "message" in df.columns:
    df = df[["label", "message"]].copy()

else:
    raise ValueError(
        f"Could not find label/message columns. Found: {list(df.columns)}"
    )

print("\nColumns after selection:")
print(list(df.columns))

print("\nFirst 5 rows after renaming:")
print(df.head())

# =========================================================
# 4. CLEAN THE DATA
# =========================================================

# Clean labels
df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Clean messages
df["message"] = (
    df["message"]
    .astype(str)
    .str.strip()
)

# Keep only valid labels
df = df[df["label"].isin(["ham", "spam"])]

# Remove empty messages
df = df[df["message"] != ""]

# Remove duplicate rows
before_duplicates = len(df)

df = (
    df
    .drop_duplicates(subset=["label", "message"])
    .reset_index(drop=True)
)

removed_duplicates = before_duplicates - len(df)

print("\nDataset after cleaning:")
print(df.shape)

print("\nDuplicates removed:")
print(removed_duplicates)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nClass percentages:")
print(
    df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# =========================================================
# 5. VISUALIZE CLASS DISTRIBUTION
# =========================================================

class_counts = (
    df["label"]
    .value_counts()
    .reindex(["ham", "spam"])
)

class_counts.plot(kind="bar")

plt.title("HAM vs SPAM Distribution")
plt.xlabel("Message Class")
plt.ylabel("Number of Messages")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "class_distribution.png",
    dpi=200,
)

plt.close()

print("\nSaved class distribution graph.")

# =========================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING SETS
# =========================================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining messages:")
print(len(X_train))

print("\nTesting messages:")
print(len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# =========================================================
# 7. CONVERT TEXT INTO NUMBERS WITH TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion complete.")

print("\nTraining feature shape:")
print(X_train_tfidf.shape)

print("\nTesting feature shape:")
print(X_test_tfidf.shape)

print("\nVocabulary size:")
print(len(vectorizer.vocabulary_))

# =========================================================
# 8. CREATE THE TWO MACHINE-LEARNING MODELS
# =========================================================

naive_bayes = MultinomialNB(
    alpha=0.3,
    fit_prior=False,
)

logistic_regression = LogisticRegression(
    max_iter=1500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    C=2.0,
)

print("\nModels created:")
print("1. Multinomial Naive Bayes")
print("2. Logistic Regression")

# =========================================================
# 9. TRAIN BOTH MODELS AND MAKE PREDICTIONS
# =========================================================

# Train Multinomial Naive Bayes
naive_bayes.fit(
    X_train_tfidf,
    y_train,
)

# Train Logistic Regression
logistic_regression.fit(
    X_train_tfidf,
    y_train,
)

print("\nBoth models trained successfully.")


# Make predictions on unseen test data
nb_predictions = naive_bayes.predict(
    X_test_tfidf
)

lr_predictions = logistic_regression.predict(
    X_test_tfidf
)

print("\nPredictions created successfully.")

# =========================================================
# 10. EVALUATE AND COMPARE BOTH MODELS
# =========================================================

# Naive Bayes metrics
nb_accuracy = accuracy_score(
    y_test,
    nb_predictions,
)

nb_precision = precision_score(
    y_test,
    nb_predictions,
    pos_label="spam",
    zero_division=0,
)

nb_recall = recall_score(
    y_test,
    nb_predictions,
    pos_label="spam",
    zero_division=0,
)

nb_f1 = f1_score(
    y_test,
    nb_predictions,
    pos_label="spam",
    zero_division=0,
)


# Logistic Regression metrics
lr_accuracy = accuracy_score(
    y_test,
    lr_predictions,
)

lr_precision = precision_score(
    y_test,
    lr_predictions,
    pos_label="spam",
    zero_division=0,
)

lr_recall = recall_score(
    y_test,
    lr_predictions,
    pos_label="spam",
    zero_division=0,
)

lr_f1 = f1_score(
    y_test,
    lr_predictions,
    pos_label="spam",
    zero_division=0,
)

comparison = pd.DataFrame(
    [
        {
            "model": "Multinomial Naive Bayes",
            "accuracy": nb_accuracy,
            "precision_spam": nb_precision,
            "recall_spam": nb_recall,
            "f1_spam": nb_f1,
        },
        {
            "model": "Logistic Regression",
            "accuracy": lr_accuracy,
            "precision_spam": lr_precision,
            "recall_spam": lr_recall,
            "f1_spam": lr_f1,
        },
    ]
)

print("\nModel comparison:")
print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# =========================================================
# 11. CREATE CONFUSION MATRICES
# =========================================================

# Naive Bayes confusion matrix
nb_cm = confusion_matrix(
    y_test,
    nb_predictions,
    labels=["ham", "spam"],
)

nb_display = ConfusionMatrixDisplay(
    confusion_matrix=nb_cm,
    display_labels=["HAM", "SPAM"],
)

nb_display.plot(
    cmap="Blues",
    values_format="d",
)

plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "confusion_matrix_naive_bayes.png",
    dpi=200,
)

plt.close()


# Logistic Regression confusion matrix
lr_cm = confusion_matrix(
    y_test,
    lr_predictions,
    labels=["ham", "spam"],
)

lr_display = ConfusionMatrixDisplay(
    confusion_matrix=lr_cm,
    display_labels=["HAM", "SPAM"],
)

lr_display.plot(
    cmap="Blues",
    values_format="d",
)

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "confusion_matrix_logistic_regression.png",
    dpi=200,
)

plt.close()

print("\nConfusion matrices saved successfully.")

# =========================================================
# 12. SAVE MODEL COMPARISON RESULTS
# =========================================================

comparison.to_csv(
    OUTPUTS_DIR / "model_comparison.csv",
    index=False,
)

print("\nModel comparison CSV saved.")

graph_data = comparison.set_index("model")[
    [
        "accuracy",
        "precision_spam",
        "recall_spam",
        "f1_spam",
    ]
]

graph_data.plot(
    kind="bar",
    figsize=(10, 6),
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "model_comparison.png",
    dpi=200,
)

plt.close()

print("Model comparison graph saved.")

best_model = comparison.sort_values(
    by=["f1_spam", "recall_spam"],
    ascending=False,
).iloc[0]

print("\nBest model:")
print(best_model["model"])

print("\nBest model Spam F1:")
print(round(best_model["f1_spam"], 4))

# =========================================================
# 13. SAVE THE VECTORIZER AND BOTH TRAINED MODELS
# =========================================================

joblib.dump(
    vectorizer,
    MODELS_DIR / "vectorizer.joblib",
)

joblib.dump(
    naive_bayes,
    MODELS_DIR / "multinomial_naive_bayes.joblib",
)

joblib.dump(
    logistic_regression,
    MODELS_DIR / "logistic_regression.joblib",
)

print("\nSaved trained files:")
print("- vectorizer.joblib")
print("- multinomial_naive_bayes.joblib")
print("- logistic_regression.joblib")

# =========================================================
# 14. FINAL TRAINING SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Dataset: {DATASET_PATH.name}")
print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")

print(f"\nBest model: {best_model['model']}")
print(f"Best Spam F1: {best_model['f1_spam']:.4f}")

print("\nSaved model files:")
print("- models/vectorizer.joblib")
print("- models/multinomial_naive_bayes.joblib")
print("- models/logistic_regression.joblib")

print("\nSaved output files:")
print("- outputs/class_distribution.png")
print("- outputs/confusion_matrix_naive_bayes.png")
print("- outputs/confusion_matrix_logistic_regression.png")
print("- outputs/model_comparison.csv")
print("- outputs/model_comparison.png")

print("\nTraining pipeline finished successfully.")