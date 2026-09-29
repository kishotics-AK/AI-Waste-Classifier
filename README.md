# ♻️ AI-Powered Waste Classifier

An image classification web app that uses **PyTorch** and **MobileNetV3
Small** to identify common types of waste and provide basic disposal
guidance.

Upload a photo of a waste item, and the app predicts one of six
categories, displays its confidence score, and shows the top three
predictions.

> **Project status:** Portfolio / educational project. Predictions may
> be inaccurate, especially for real-world images that differ from the
> training dataset. Always follow local waste-disposal rules.

------------------------------------------------------------------------

## ✨ Features

-   Upload waste images in **JPG, JPEG, or PNG** format.
-   Classify images into six categories:
    -   Cardboard
    -   Glass
    -   Metal
    -   Paper
    -   Plastic
    -   Trash
-   View the predicted category and confidence score.
-   Explore the top three model predictions.
-   Get general disposal guidance for the predicted category.
-   Run the app locally through a simple Streamlit interface.

## 🧠 Model and Technology

  Component                 Details
  ------------------------- ----------------------
  Model architecture        MobileNetV3 Small
  Deep learning framework   PyTorch
  Image input size          224 × 224 pixels
  Number of classes         6
  Dataset                   TrashNet
  Web interface             Streamlit
  Image processing          torchvision / Pillow
  Compute                   CPU-compatible

The V2 model uses pretrained MobileNetV3 Small features with fine-tuning
of the final feature layers and a six-class classification head.
Class-weighted cross-entropy loss was used to help address class
imbalance.

## 📊 Model Performance

The V2 model was evaluated on a held-out test split from TrashNet.

  Metric                   Score
  ----------------- ------------
  Test accuracy       **81.25%**
  Macro precision         79.52%
  Macro recall            83.02%
  Macro F1-score          80.28%

### Per-class results

  Class         Precision   Recall   F1-score
  ----------- ----------- -------- ----------
  Cardboard        91.38%   86.89%     89.08%
  Glass            75.00%   78.95%     76.92%
  Metal            74.63%   80.65%     77.52%
  Paper            93.67%   82.22%     87.57%
  Plastic          85.71%   73.97%     79.41%
  Trash            56.76%   95.45%     71.19%

A separate small real-world image set produced **35.48% accuracy**
during an informal evaluation. This illustrates the gap between the
curated test set and photos taken in varied real-world conditions.
Results depend on image quality, background, lighting, object
appearance, and dataset similarity.

## 🗂️ Project Structure

``` text
AI-Waste-Classifier/
├── app.py
├── README.md
├── data/
│   ├── raw/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── real_world/
├── models/
│   ├── waste_classifier.pth
│   └── waste_classifier_v2.pth
├── notebooks/
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── train_v2.py
│   ├── evaluate.py
│   ├── evaluate_realworld.py
│   └── predict.py
└── assets/
```

> The trained `.pth` model files and image datasets may be too large to
> include directly in a Git repository. If they are not included,
> download or generate the required files separately and place them in
> the paths expected by the scripts.

## ⚙️ Setup and Installation

### 1. Prerequisites

-   Windows, macOS, or Linux
-   Python 3.14 or a compatible Python version supported by the
    installed PyTorch and Streamlit packages
-   Git (optional, for cloning the repository)

### 2. Clone the repository

``` bash
git clone <your-repository-url>
cd AI-Waste-Classifier
```

If you already have the project folder, open a terminal in that
directory instead.

### 3. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, you can run the environment's
Python directly without activating it.

**macOS / Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

Install the packages used by the project:

``` bash
python -m pip install --upgrade pip
python -m pip install torch torchvision pillow pandas streamlit
```

For a reproducible setup, consider adding a project-specific
`requirements.txt` with the exact package versions used in your
environment.

## ▶️ Run the Web App

From the project root:

``` bash
python -m streamlit run app.py
```

On Windows, if you are using the project's virtual environment without
activating it:

``` powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that address in
your browser, upload an image, and select **Classify Waste**.

## 🏋️ Training and Evaluation

The repository includes scripts for preparing the dataset, training the
model, and evaluating predictions.

### Prepare the dataset

``` bash
python src/prepare_dataset.py
```

### Train the original model

``` bash
python src/train.py
```

### Train V2

``` bash
python src/train_v2.py
```

### Evaluate the model

``` bash
python src/evaluate.py
```

### Evaluate on the real-world image set

``` bash
python src/evaluate_realworld.py
```

Check each script's configuration and expected data/model paths before
running it. Training requires the dataset to be present in the expected
directory structure.

## 🖼️ Predict an Image from the Command Line

Run:

``` bash
python src/predict.py
```

When prompted, enter the full path to an image file. The script displays
the top predictions and their confidence scores.

## 🗃️ Dataset

This project uses **TrashNet**, a waste-image dataset containing six
categories: cardboard, glass, metal, paper, plastic, and trash.

The original dataset contains **2,527 images**. The project uses a 70% /
15% / 15% train-validation-test split:

  Split             Images
  ------------ -----------
  Training           1,766
  Validation           377
  Test                 384
  **Total**      **2,527**

Dataset source: [TrashNet on
GitHub](https://github.com/garythung/trashnet).

Please review and follow the dataset's license and usage terms before
redistributing the data.

## ⚠️ Limitations

-   The model is trained on a relatively small, curated dataset.
-   Test-set performance does not guarantee accuracy on everyday photos.
-   Some categories can look similar, such as glass and plastic or paper
    and cardboard.
-   Confidence scores are model outputs, not guarantees that a
    prediction is correct.
-   Disposal guidance is general and may not match local recycling
    regulations.
-   This project is intended for learning and demonstration, not for
    safety-critical or commercial waste-sorting decisions.

## 🔮 Possible Improvements

-   Expand the training data with diverse real-world images.
-   Add a confidence threshold and an "uncertain prediction" message.
-   Improve model evaluation with more varied images and per-class
    analysis.
-   Add image-quality checks and clearer guidance for ambiguous items.
-   Deploy the app to a hosting platform after reviewing model and
    dataset licensing.

## 👨‍💻 Author

**Kishore N.**\
BE Computer Science and Engineering \| Aspiring AI/ML Engineer

------------------------------------------------------------------------

*Built as an educational portfolio project using PyTorch, MobileNetV3
Small, and Streamlit.*
