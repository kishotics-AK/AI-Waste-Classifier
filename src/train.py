import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------------
# 1. Paths and settings
# -----------------------------------

TRAIN_DIR = r"D:\Project\AI-Waste-Classifier\data\train"
VAL_DIR = r"D:\Project\AI-Waste-Classifier\data\validation"
MODEL_DIR = r"D:\Project\AI-Waste-Classifier\models"

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# -----------------------------------
# 2. Image transformations
# -----------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# 3. Load datasets
# -----------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------------
# 4. Create DataLoaders
# -----------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------------
# 5. Load pretrained MobileNetV3
# -----------------------------------

weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

# Freeze the pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# -----------------------------------
# 6. Replace the final classifier
# -----------------------------------

number_of_classes = len(train_dataset.classes)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    number_of_classes
)

model = model.to(DEVICE)


# -----------------------------------
# 7. Loss and optimizer
# -----------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[3].parameters(),
    lr=LEARNING_RATE
)


# -----------------------------------
# 8. Training
# -----------------------------------

best_val_accuracy = 0.0

os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -----------------------------------
    # Validation
    # -----------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes
            },
            os.path.join(MODEL_DIR, "waste_classifier.pth")
        )

        print("Best model saved!")


print("\nTraining completed!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")