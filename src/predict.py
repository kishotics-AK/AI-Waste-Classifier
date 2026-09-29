import torch
from torchvision import models, transforms
from PIL import Image
import os

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = r"D:\Project\AI-Waste-Classifier\models\waste_classifier_v2.pth"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_SIZE = 224

# ==============================
# DEVICE
# ==============================

device = torch.device("cpu")

print("Using device:", device)

# ==============================
# IMAGE PREPROCESSING
# ==============================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# LOAD MODEL
# ==============================

model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    len(CLASSES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully!")

# ==============================
# PREDICTION
# ==============================

def predict_image(image_path):

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

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

    print("\n===================================")
    print("AI WASTE CLASSIFICATION")
    print("===================================")

    print("Image:", image_path)

    print("\nTop 3 Predictions:")

    for i in range(3):

        predicted_class = CLASSES[
            top_indices[0][i].item()
        ]

        confidence = (
            top_probabilities[0][i].item()
            * 100
        )

        print(
            f"{i + 1}. "
            f"{predicted_class.upper():<10} "
            f"{confidence:.2f}%"
        )

    print("===================================")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":

    image_path = input(
        "\nEnter the full path of a waste image: "
    ).strip('"')

    predict_image(image_path)