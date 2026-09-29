import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"D:\Project\AI-Waste-Classifier\models\waste_classifier_v2.pth"

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

    model = models.mobilenet_v3_small(weights=None)

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
# USER INTERFACE
# ==========================================

st.title("♻️ AI Waste Classifier")

st.write(
    "Upload an image of waste and let the AI "
    "classify it into one of six categories."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    if st.button(
        "🔍 Classify Waste",
        use_container_width=True
    ):

        predictions = predict_image(image)

        top_class = predictions[0][0]
        top_confidence = predictions[0][1]

        st.success(
            f"Prediction: {top_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{top_confidence:.2f}%"
        )

        # ----------------------------------
        # DISPOSAL GUIDANCE
        # ----------------------------------

        st.subheader("♻️ Disposal Guidance")

        st.info(
            DISPOSAL_GUIDANCE[top_class]
        )

        # ----------------------------------
        # TOP 3 PREDICTIONS
        # ----------------------------------

        st.subheader("📊 Top 3 Predictions")

        for class_name, confidence in predictions:

            st.write(
                f"**{class_name.capitalize()}** — "
                f"{confidence:.2f}%"
            )

            st.progress(
                min(int(confidence), 100)
            )

st.divider()

st.caption(
    "AI Waste Classifier • PyTorch + MobileNetV3"
)