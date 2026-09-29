import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------------
# 1. Paths
# -----------------------------------

TEST_DIR = r"D:\Project\AI-Waste-Classifier\data\test"
MODEL_PATH = r"D:\Project\AI-Waste-Classifier\models\waste_classifier_v2.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# -----------------------------------
# 2. Image preprocessing
# -----------------------------------

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# 3. Load test dataset
# -----------------------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

classes = test_dataset.classes

print("Classes:", classes)
print("Test images:", len(test_dataset))


# -----------------------------------
# 4. Load trained model
# -----------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = models.mobilenet_v3_small(weights=None)

number_of_classes = len(classes)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    number_of_classes
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()


# -----------------------------------
# 5. Make predictions
# -----------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


# -----------------------------------
# 6. Create confusion matrix
# -----------------------------------

num_classes = len(classes)

confusion = [
    [0 for _ in range(num_classes)]
    for _ in range(num_classes)
]

for actual, predicted in zip(all_labels, all_predictions):
    confusion[actual][predicted] += 1


# -----------------------------------
# 7. Overall accuracy
# -----------------------------------

correct = sum(
    confusion[i][i]
    for i in range(num_classes)
)

total = len(all_labels)

accuracy = correct / total

print("\n===================================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("===================================")


# -----------------------------------
# 8. Calculate metrics for each class
# -----------------------------------

print("\nClassification Report:")
print()

print(
    f"{'Class':<12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1-Score':>12}"
    f"{'Support':>10}"
)

print("-" * 58)

precisions = []
recalls = []
f1_scores = []

for i, class_name in enumerate(classes):

    true_positive = confusion[i][i]

    false_positive = sum(
        confusion[row][i]
        for row in range(num_classes)
        if row != i
    )

    false_negative = sum(
        confusion[i][column]
        for column in range(num_classes)
        if column != i
    )

    support = sum(confusion[i])

    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(
        f"{class_name:<12}"
        f"{precision:>12.4f}"
        f"{recall:>12.4f}"
        f"{f1:>12.4f}"
        f"{support:>10}"
    )


# -----------------------------------
# 9. Macro averages
# -----------------------------------

macro_precision = sum(precisions) / num_classes
macro_recall = sum(recalls) / num_classes
macro_f1 = sum(f1_scores) / num_classes

print("-" * 58)

print(
    f"{'Macro Avg':<12}"
    f"{macro_precision:>12.4f}"
    f"{macro_recall:>12.4f}"
    f"{macro_f1:>12.4f}"
    f"{total:>10}"
)


# -----------------------------------
# 10. Confusion matrix
# -----------------------------------

print("\nConfusion Matrix:")
print()

print("Actual ↓ / Predicted →")
print(f"{'':<12}", end="")

for class_name in classes:
    print(f"{class_name[:10]:>12}", end="")

print()

for i, class_name in enumerate(classes):

    print(f"{class_name:<12}", end="")

    for j in range(num_classes):
        print(f"{confusion[i][j]:>12}", end="")

    print()


print("\nEvaluation completed successfully!")