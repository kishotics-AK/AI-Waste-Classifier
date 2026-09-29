import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WasteAI | Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f7f9fc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* HERO */

.hero {
    text-align: center;
    padding: 30px 10px 35px 10px;
}

.hero-icon {
    font-size: 55px;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #111827;
    margin-top: 5px;
}

.hero-title span {
    color: #16a34a;
}

.hero-subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-top: 8px;
}

/* CARDS */

.card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 25px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
}

.section-description {
    font-size: 15px;
    color: #6b7280;
}

/* PREDICTION */

.prediction-card {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #f0fdf4
    );

    border: 1px solid #bbf7d0;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.prediction-label {
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-name {
    font-size: 38px;
    font-weight: 800;
    color: #15803d;
    margin: 5px 0;
}

.confidence {
    font-size: 17px;
    color: #374151;
}

/* INFO CARDS */

.info-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 25px;
    min-height: 120px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.04);
}

/* BUTTON */

div.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 12px;
    background-color: #16a34a;
    color: white;
    border: none;
    font-size: 16px;
    font-weight: 700;
}

div.stButton > button:hover {
    background-color: #15803d;
    color: white;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 14px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "waste_classifier_v2.pth"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# ==========================================
# DISPOSAL GUIDANCE
# ==========================================

DISPOSAL_GUIDANCE = {

    "cardboard":
        "Flatten the cardboard and place it in the appropriate paper/cardboard recycling collection.",

    "glass":
        "Place glass items in a designated glass recycling container. Handle broken glass carefully.",

    "metal":
        "Clean metal cans or containers and place them in the appropriate metal recycling collection.",

    "paper":
        "Keep paper clean and dry and place it in the appropriate paper recycling collection.",

    "plastic":
        "Empty and rinse plastic containers when appropriate, then place them in the appropriate plastic recycling collection.",

    "trash":
        "Dispose of this item in the appropriate general-waste collection according to local waste rules."
}


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = models.mobilenet_v3_small(
        weights=None
    )

    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        len(CLASSES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


model = load_model()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            3,
            dim=1
        )

    predictions = []

    for i in range(3):

        class_name = CLASSES[
            top_indices[0][i].item()
        ]

        confidence = (
            top_probabilities[0][i].item()
            * 100
        )

        predictions.append(
            (class_name, confidence)
        )

    return predictions


# ==========================================
# HERO
# ==========================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">♻️</div>

    <div class="hero-title">
        Waste<span>AI</span>
    </div>

    <div class="hero-subtitle">
        Intelligent waste classification powered by deep learning
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# INTRODUCTION
# ==========================================

st.markdown("""
<div class="card">

    <div class="section-title">
        🔍 Identify Your Waste
    </div>

    <div class="section-description">
        Upload an image of a waste item and our AI model
        will classify it into one of six waste categories.
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload a waste image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)


# ==========================================
# IMAGE PROCESSING
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    # ======================================
    # IMAGE PREVIEW
    # ======================================

    with col1:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                🖼️ Uploaded Image
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )


    # ======================================
    # AI ANALYSIS
    # ======================================

    with col2:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                🤖 AI Analysis
            </div>

            <div class="section-description">
                MobileNetV3 analyzes the uploaded image
                and predicts the waste category.
            </div>

        </div>
        """, unsafe_allow_html=True)

        classify_button = st.button(
            "🔍 Analyze Waste",
            use_container_width=True
        )

        if classify_button:

            with st.spinner(
                "Analyzing image..."
            ):

                predictions = predict_image(
                    image
                )

            top_class = predictions[0][0]

            top_confidence = predictions[0][1]


            # ==================================
            # MAIN PREDICTION
            # ==================================

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        Predicted Category
                    </div>

                    <div class="prediction-name">
                        {top_class.upper()}
                    </div>

                    <div class="confidence">
                        Confidence:
                        <strong>
                            {top_confidence:.2f}%
                        </strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ==================================
            # DISPOSAL GUIDANCE
            # ==================================

            st.markdown("""
            <div class="card">

                <div class="section-title">
                    ♻️ Disposal Guidance
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.info(
                DISPOSAL_GUIDANCE[top_class]
            )


            # ==================================
            # TOP 3 PREDICTIONS
            # ==================================

            st.markdown("""
            <div class="card">

                <div class="section-title">
                    📊 Top 3 Predictions
                </div>

            </div>
            """, unsafe_allow_html=True)

            for rank, (
                class_name,
                confidence
            ) in enumerate(
                predictions,
                start=1
            ):

                st.write(
                    f"**#{rank} "
                    f"{class_name.capitalize()}** — "
                    f"{confidence:.2f}%"
                )

                st.progress(
                    min(int(confidence), 100)
                )


# ==========================================
# MODEL INFORMATION
# ==========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""
    <div class="info-card">

        <div class="section-title">
            🧠 Model
        </div>

        <p>
            MobileNetV3 Small
        </p>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="info-card">

        <div class="section-title">
            🗂️ Categories
        </div>

        <p>
            6 waste categories
        </p>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="info-card">

        <div class="section-title">
            ⚡ Framework
        </div>

        <p>
            PyTorch + Streamlit
        </p>

    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    AI Waste Classifier • Built with PyTorch + MobileNetV3 + Streamlit

</div>
""", unsafe_allow_html=True)
