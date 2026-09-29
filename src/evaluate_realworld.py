import os
import torch
from torchvision import models, transforms
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = r"D:\Project\AI-Waste-Classifier"

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "real_world"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "waste_classifier_v2.pth"
)

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

device = torch.device("cpu")

# ==============================
# IMAGE TRANSFORMATION
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
# EVALUATION
# ==============================

correct = 0
total = 0

confusion_matrix = [
    [0 for _ in CLASSES]
    for _ in CLASSES
]

print("\nREAL-WORLD PREDICTIONS")
print("=" * 75)

for actual_index, class_name in enumerate(CLASSES):

    class_folder = os.path.join(
        DATASET_PATH, class_name
    )

    if not os.path.isdir(class_folder):
        print(f"Missing folder: {class_folder}")
        continue

    for filename in os.listdir(class_folder):

        image_path = os.path.join(
            class_folder, filename
        )

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted_index = torch.max(
                    probabilities, dim=1
                )

            predicted_index = predicted_index.item()
            confidence = confidence.item() * 100

            total += 1

            if predicted_index == actual_index:
                correct += 1

            confusion_matrix[actual_index][predicted_index] += 1

            status = (
                "CORRECT"
                if predicted_index == actual_index
                else "WRONG"
            )

            print(
                f"Actual: {class_name:<10} "
                f"Predicted: {CLASSES[predicted_index]:<10} "
                f"Confidence: {confidence:6.2f}% "
                f"{status}"
            )

        except Exception as error:
            print(f"Could not process {filename}: {error}")

# ==============================
# RESULTS
# ==============================

print("\n" + "=" * 75)
print("REAL-WORLD EVALUATION RESULTS")
print("=" * 75)

if total > 0:
    accuracy = (correct / total) * 100

    print(f"Total Images: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Real-World Accuracy: {accuracy:.2f}%")

print("\nCONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted")

print(
    f"{'Actual':<12}"
    + "".join(f"{name[:9]:>11}" for name in CLASSES)
)

for i, class_name in enumerate(CLASSES):
    print(
        f"{class_name:<12}"
        + "".join(f"{value:>11}" for value in confusion_matrix[i])
    )

print("\nEvaluation completed!")