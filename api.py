from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "waste_classifier_v2.pth"
)


CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# =========================================================
# DISPOSAL GUIDANCE
# =========================================================

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


# =========================================================
# IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading AI model...")

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

print("AI model loaded successfully.")


# =========================================================
# CREATE FASTAPI APP
# =========================================================

app = FastAPI(
    title="WasteAI API",
    description="AI Waste Classification API",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "WasteAI API is running",
        "model": "MobileNetV3 Small",
        "classes": CLASSES
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image_bytes = await file.read()

    image = Image.open(
        __import__("io").BytesIO(image_bytes)
    ).convert("RGB")


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)


    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Top 3 predictions

        top_probabilities, top_indices = torch.topk(
            probabilities,
            3,
            dim=1
        )


    # -----------------------------------------------------
    # Build prediction list
    # -----------------------------------------------------

    predictions = []


    for i in range(3):

        class_name = CLASSES[
            top_indices[0][i].item()
        ]


        confidence = (
            top_probabilities[0][i].item()
            * 100
        )


        predictions.append({

            "class": class_name,

            "confidence": round(
                confidence,
                2
            )

        })


    # -----------------------------------------------------
    # Main prediction
    # -----------------------------------------------------

    top_class = predictions[0]["class"]

    top_confidence = predictions[0]["confidence"]


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return {

        "success": True,

        "filename": file.filename,

        "class": top_class,

        "confidence": top_confidence,

        "disposal_guidance":
            DISPOSAL_GUIDANCE[top_class],

        "top_predictions":
            predictions

    }
