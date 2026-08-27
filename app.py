from pathlib import Path

import joblib
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered",
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
NB_MODEL_PATH = MODELS_DIR / "multinomial_naive_bayes.joblib"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>

/* ---------- App background ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% 0%, rgba(76, 92, 210, 0.12), transparent 26%),
        radial-gradient(circle at 88% 6%, rgba(139, 92, 246, 0.12), transparent 26%),
        linear-gradient(180deg, #080d18 0%, #0a1220 100%);
}

/* ---------- Streamlit top bar ---------- */
[data-testid="stHeader"] {
    background: rgba(8, 13, 24, 0.92);
    border-bottom: 1px solid rgba(128, 148, 196, 0.08);
}

/* ---------- Main content ---------- */
[data-testid="stMainBlockContainer"] {
    max-width: 980px;
    padding-top: 4.25rem;
    padding-bottom: 2.75rem;
}

.block-container {
    max-width: 980px !important;
    padding-top: 4.25rem !important;
    padding-bottom: 2.75rem !important;
}

/* ---------- Hero ---------- */
.hero-card {
    position: relative;
    overflow: hidden;
    padding: 1.75rem 1.85rem 1.7rem 1.85rem;
    margin-bottom: 1.35rem;
    border: 1px solid rgba(126, 145, 192, 0.22);
    border-radius: 22px;
    background: linear-gradient(
        145deg,
        rgba(21, 31, 53, 0.96),
        rgba(13, 22, 39, 0.96)
    );
    box-shadow: 0 18px 54px rgba(0, 0, 0, 0.20);
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -105px;
    top: -120px;
    border-radius: 50%;
    background: rgba(117, 91, 255, 0.17);
    filter: blur(12px);
    pointer-events: none;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.9rem;
    padding: 0.36rem 0.68rem;
    border-radius: 999px;
    border: 1px solid rgba(131, 111, 255, 0.34);
    background: rgba(105, 85, 241, 0.10);
    color: #c1b9ff;
    font-size: 0.75rem;
    font-weight: 760;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.hero-kicker-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7697ff, #a66cff);
    box-shadow: 0 0 14px rgba(135, 107, 255, 0.8);
}

.hero-title {
    margin: 0;
    color: #f7f9ff;
    font-size: 2.55rem;
    line-height: 1.08;
    font-weight: 820;
    letter-spacing: -0.035em;
}

.hero-title-accent {
    background: linear-gradient(90deg, #7fa3ff 0%, #8f75ff 52%, #c172ff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
}

.hero-copy {
    max-width: 730px;
    margin-top: 0.8rem;
    color: #acbad2;
    font-size: 0.98rem;
    line-height: 1.65;
}

/* ---------- Section heading ---------- */
.section-heading {
    margin: 0.35rem 0 0.55rem 0;
    color: #f5f7ff;
    font-size: 1rem;
    font-weight: 760;
}

.section-help {
    margin-bottom: 0.75rem;
    color: #7f90ae;
    font-size: 0.84rem;
}

/* ---------- Text area ---------- */
[data-testid="stTextArea"] {
    margin-bottom: 0.55rem;
}

[data-testid="stTextArea"] label {
    display: none !important;
}

[data-testid="stTextArea"] div[data-baseweb="textarea"] {
    background: #111a2b !important;
    border: 1px solid #30415f !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    overflow: hidden;
}

[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
    border-color: #7569ff !important;
    box-shadow: 0 0 0 2px rgba(117, 105, 255, 0.14) !important;
}

[data-testid="stTextArea"] textarea {
    min-height: 126px !important;
    padding: 15px 16px !important;
    background: transparent !important;
    color: #f8faff !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-size: 0.97rem !important;
    line-height: 1.52 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #71819f !important;
}

/* ---------- Button ---------- */
[data-testid="stButton"] {
    margin-top: 0.2rem;
}

[data-testid="stButton"] button {
    min-height: 50px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    background: linear-gradient(
        90deg,
        #4f70ef 0%,
        #675cf2 52%,
        #8d58ef 100%
    ) !important;
    color: #ffffff !important;
    font-size: 0.96rem !important;
    font-weight: 760 !important;
    box-shadow: 0 10px 28px rgba(82, 74, 210, 0.18);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}

[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 32px rgba(82, 74, 210, 0.28);
    color: #ffffff !important;
}

[data-testid="stButton"] button:focus,
[data-testid="stButton"] button:focus-visible {
    outline: none !important;
    border-color: rgba(166, 142, 255, 0.85) !important;
    box-shadow: 0 0 0 2px rgba(125, 105, 255, 0.18) !important;
}

/* ---------- Results heading ---------- */
.results-heading {
    margin: 1.5rem 0 0.8rem 0;
    color: #f4f7ff;
    font-size: 1.03rem;
    font-weight: 780;
}

/* ---------- Model cards ---------- */
.model-card {
    min-height: 260px;
    height: 100%;
    padding: 1.35rem;
    border: 1px solid rgba(126, 145, 192, 0.22);
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        rgba(23, 33, 56, 0.97),
        rgba(16, 25, 44, 0.97)
    );
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.16);
}

.model-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.model-name {
    color: #f8faff;
    font-size: 1.04rem;
    font-weight: 780;
}

.model-type {
    flex: 0 0 auto;
    padding: 0.28rem 0.52rem;
    border-radius: 999px;
    border: 1px solid rgba(102, 124, 170, 0.24);
    background: rgba(82, 103, 145, 0.13);
    color: #93a7ca;
    font-size: 0.70rem;
    font-weight: 720;
}

.prediction-badge {
    padding: 0.86rem 0.95rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.02rem;
    font-weight: 800;
}

.prediction-badge.ham {
    color: #aaf3d1;
    background: linear-gradient(
        90deg,
        rgba(13, 107, 80, 0.62),
        rgba(13, 76, 68, 0.62)
    );
    border: 1px solid rgba(48, 190, 139, 0.28);
}

.prediction-badge.spam {
    color: #ffd0d7;
    background: linear-gradient(
        90deg,
        rgba(132, 43, 67, 0.64),
        rgba(91, 32, 51, 0.64)
    );
    border: 1px solid rgba(242, 89, 122, 0.28);
}

.confidence-block {
    margin-top: 1.15rem;
}

.confidence-label {
    color: #8fa1bf;
    font-size: 0.80rem;
    margin-bottom: 0.28rem;
}

.confidence-value {
    color: #eef3ff;
    font-size: 2.1rem;
    line-height: 1;
    font-weight: 830;
    letter-spacing: -0.02em;
}

.confidence-bar {
    width: 100%;
    height: 7px;
    margin-top: 0.8rem;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(91, 108, 143, 0.22);
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #5d7cff, #8f67ff);
}

.confidence-caption {
    margin-top: 0.55rem;
    color: #657793;
    font-size: 0.73rem;
}

/* ---------- Summary banner ---------- */
.summary-banner {
    margin-top: 0.9rem;
    padding: 0.92rem 1rem;
    border-radius: 13px;
    text-align: center;
    font-size: 0.92rem;
    font-weight: 760;
}

.summary-banner.ham {
    color: #baf6d9;
    background: linear-gradient(
        90deg,
        rgba(14, 101, 76, 0.60),
        rgba(25, 73, 86, 0.60)
    );
    border: 1px solid rgba(50, 184, 134, 0.28);
}

.summary-banner.spam {
    color: #ffd0d7;
    background: linear-gradient(
        90deg,
        rgba(125, 43, 62, 0.63),
        rgba(82, 33, 55, 0.63)
    );
    border: 1px solid rgba(235, 85, 118, 0.26);
}

.summary-banner.disagree {
    color: #ffe1aa;
    background: linear-gradient(
        90deg,
        rgba(103, 69, 27, 0.63),
        rgba(85, 49, 41, 0.63)
    );
    border: 1px solid rgba(218, 157, 64, 0.28);
}

/* ---------- Footer note ---------- */
.footer-note {
    margin-top: 1rem;
    text-align: center;
    color: #62718c;
    font-size: 0.75rem;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- Mobile ---------- */
@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 3.5rem !important;
    }

    .hero-card {
        padding: 1.35rem;
        border-radius: 18px;
    }

    .hero-title {
        font-size: 2rem;
    }

    .hero-copy {
        font-size: 0.92rem;
    }

    .model-card {
        min-height: auto;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================
hero_html = (
    '<div class="hero-card">'
    '<div class="hero-kicker">'
    '<span class="hero-kicker-dot"></span>'
    'AI / ML Fellowship Project'
    '</div>'
    '<h1 class="hero-title">'
    'Spam Message <span class="hero-title-accent">Detector</span>'
    '</h1>'
    '<div class="hero-copy">'
    'Compare two classical machine-learning models on the same SMS. '
    'The message is transformed with the saved TF-IDF vectorizer, then '
    'classified by Multinomial Naive Bayes and Logistic Regression.'
    '</div>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# =========================================================
# CHECK TRAINED FILES
# =========================================================
required_files = [
    VECTORIZER_PATH,
    NB_MODEL_PATH,
    LR_MODEL_PATH,
]

missing_files = [
    file_path
    for file_path in required_files
    if not file_path.exists()
]

if missing_files:
    st.error(
        "Some trained files are missing. "
        "Run `python train_models.py` first."
    )

    for file_path in missing_files:
        st.write(f"Missing: `{file_path.name}`")

    st.stop()


# =========================================================
# LOAD TRAINED OBJECTS
# =========================================================
@st.cache_resource
def load_trained_objects():
    vectorizer = joblib.load(VECTORIZER_PATH)
    naive_bayes = joblib.load(NB_MODEL_PATH)
    logistic_regression = joblib.load(LR_MODEL_PATH)

    return (
        vectorizer,
        naive_bayes,
        logistic_regression,
    )


vectorizer, naive_bayes, logistic_regression = load_trained_objects()


# =========================================================
# PREDICTION HELPERS
# =========================================================
def predict_with_confidence(
    model,
    message_features,
):
    prediction = model.predict(
        message_features
    )[0]

    probabilities = model.predict_proba(
        message_features
    )[0]

    prediction_index = list(
        model.classes_
    ).index(prediction)

    confidence = (
        probabilities[prediction_index]
        * 100
    )

    return prediction, confidence


def render_model_card(
    model_name,
    model_type,
    prediction,
    confidence,
):
    result_class = (
        "spam"
        if prediction == "spam"
        else "ham"
    )

    icon = (
        "🚨"
        if prediction == "spam"
        else "✓"
    )

    confidence_width = max(
        0,
        min(100, confidence),
    )

    card_html = (
        '<div class="model-card">'
        '<div class="model-card-top">'
        f'<div class="model-name">{model_name}</div>'
        f'<div class="model-type">{model_type}</div>'
        '</div>'
        f'<div class="prediction-badge {result_class}">'
        f'{icon} Prediction: {prediction.upper()}'
        '</div>'
        '<div class="confidence-block">'
        '<div class="confidence-label">Model confidence</div>'
        f'<div class="confidence-value">{confidence:.1f}%</div>'
        '<div class="confidence-bar">'
        f'<div class="confidence-bar-fill" style="width:{confidence_width:.1f}%"></div>'
        '</div>'
        '<div class="confidence-caption">'
        'Probability assigned to the predicted class'
        '</div>'
        '</div>'
        '</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


# =========================================================
# INPUT AREA
# =========================================================
st.markdown(
    '<div class="section-heading">Test an SMS message</div>'
    '<div class="section-help">'
    'Enter a message below and compare how both models classify it.'
    '</div>',
    unsafe_allow_html=True,
)

message = st.text_area(
    "SMS message",
    placeholder=(
        "Example: Congratulations! "
        "You have won a cash prize. Claim now."
    ),
    height=126,
    label_visibility="collapsed",
)

compare_button = st.button(
    "Compare both models",
    use_container_width=True,
)


# =========================================================
# RUN BOTH MODELS
# =========================================================
if compare_button:

    if not message.strip():

        st.warning(
            "Please enter an SMS message "
            "before comparing the models."
        )

    else:

        # Convert the user's message using the same
        # fitted TF-IDF vectorizer used during training.
        message_tfidf = vectorizer.transform(
            [message]
        )

        nb_prediction, nb_confidence = (
            predict_with_confidence(
                naive_bayes,
                message_tfidf,
            )
        )

        lr_prediction, lr_confidence = (
            predict_with_confidence(
                logistic_regression,
                message_tfidf,
            )
        )

        st.markdown(
            '<div class="results-heading">'
            'Model comparison'
            '</div>',
            unsafe_allow_html=True,
        )

        left_column, right_column = st.columns(
            2,
            gap="large",
        )

        with left_column:
            render_model_card(
                "Multinomial Naive Bayes",
                "Probabilistic",
                nb_prediction,
                nb_confidence,
            )

        with right_column:
            render_model_card(
                "Logistic Regression",
                "Linear classifier",
                lr_prediction,
                lr_confidence,
            )

        if nb_prediction == lr_prediction:

            if nb_prediction == "spam":

                summary_html = (
                    '<div class="summary-banner spam">'
                    '🚨 Both models agree: SPAM'
                    '</div>'
                )

            else:

                summary_html = (
                    '<div class="summary-banner ham">'
                    '✓ Both models agree: HAM'
                    '</div>'
                )

        else:

            summary_html = (
                '<div class="summary-banner disagree">'
                'The models disagree on this message — '
                'a useful edge case for comparison.'
                '</div>'
            )

        st.markdown(
            summary_html,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="footer-note">'
            'Confidence is the model probability for its predicted class, '
            'not a guarantee of real-world correctness.'
            '</div>',
            unsafe_allow_html=True,
        )
