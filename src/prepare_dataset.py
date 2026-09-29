import os
import random
import shutil

# -----------------------------
# 1. Paths
# -----------------------------

SOURCE_DIR = r"D:\Project\AI-Waste-Classifier\data\raw\dataset-resized"
OUTPUT_DIR = r"D:\Project\AI-Waste-Classifier\data"

# Dataset split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Makes the split reproducible
random.seed(42)

# Waste categories
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# -----------------------------
# 2. Create output directories
# -----------------------------

for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        folder = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(folder, exist_ok=True)


# -----------------------------
# 3. Split each class
# -----------------------------

for class_name in CLASSES:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        file for file in os.listdir(source_class_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Shuffle images randomly
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    print(f"\n{class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(validation_images)}")
    print(f"Test: {len(test_images)}")

    # Copy images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(OUTPUT_DIR, "train", class_name, image)
        )

    for image in validation_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(OUTPUT_DIR, "validation", class_name, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(OUTPUT_DIR, "test", class_name, image)
        )


print("\nDataset preparation completed successfully!")
