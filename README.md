# ♻️ AI-Powered Waste Classifier

An AI-powered waste classification web application built with **PyTorch**, **MobileNetV3 Small**, **FastAPI**, and a modern **HTML/CSS/JavaScript frontend**.

Upload an image of a waste item and the application predicts one of six waste categories, displays the prediction confidence, shows the top three predictions, and provides general disposal guidance.

> **Project Status:** Portfolio / educational project. Predictions may be inaccurate, especially for real-world images that differ from the training dataset. Always follow local waste-disposal rules.

---

## ✨ Features

* 🖼️ Upload waste images in JPG, JPEG, or PNG format
* 🤖 AI-based image classification using MobileNetV3 Small
* ♻️ Classifies waste into six categories:

  * Cardboard
  * Glass
  * Metal
  * Paper
  * Plastic
  * Trash
* 📊 Displays prediction confidence
* 🔍 Shows the top three model predictions
* 💡 Provides general disposal guidance
* ⚡ FastAPI backend for model inference
* 🎨 Modern responsive HTML/CSS/JavaScript frontend
* 🖥️ Streamlit interface also included
* 💻 CPU-compatible inference
* 📁 Includes training and evaluation scripts

---

## 🧠 Model & Technology

| Component               | Technology            |
| ----------------------- | --------------------- |
| Model Architecture      | MobileNetV3 Small     |
| Deep Learning Framework | PyTorch               |
| Image Processing        | Torchvision + Pillow  |
| Input Size              | 224 × 224 pixels      |
| Number of Classes       | 6                     |
| Dataset                 | TrashNet              |
| Backend API             | FastAPI               |
| Frontend                | HTML, CSS, JavaScript |
| Alternative UI          | Streamlit             |
| Inference               | CPU-compatible        |

The project uses a MobileNetV3 Small image-classification architecture with a six-class classification head.

The image preprocessing pipeline resizes input images to **224 × 224 pixels**, converts them to tensors, and applies ImageNet-style normalization before inference.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      User Image     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ HTML/CSS/JavaScript │
                    │     Frontend        │
                    └──────────┬──────────┘
                               │
                         POST /predict
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Image Preprocessing│
                    │   224 × 224 RGB     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  MobileNetV3 Small  │
                    │    PyTorch Model    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Top Predictions   │
                    │    + Confidence     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Frontend Result UI  │
                    └─────────────────────┘
```

---

## 📊 Model Performance

The V2 model was evaluated on a held-out test split from TrashNet.

| Metric          |      Score |
| --------------- | ---------: |
| Test Accuracy   | **81.25%** |
| Macro Precision |     79.52% |
| Macro Recall    |     83.02% |
| Macro F1-Score  |     80.28% |

### Per-Class Results

| Class     | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Cardboard |    91.38% | 86.89% |   89.08% |
| Glass     |    75.00% | 78.95% |   76.92% |
| Metal     |    74.63% | 80.65% |   77.52% |
| Paper     |    93.67% | 82.22% |   87.57% |
| Plastic   |    85.71% | 73.97% |   79.41% |
| Trash     |    56.76% | 95.45% |   71.19% |

### Real-World Evaluation

A separate small set of 62 real-world images produced:

**35.48% accuracy**

during an informal evaluation.

This demonstrates the difference between performance on a curated dataset and performance on varied real-world photographs.

Real-world performance can be affected by:

* Lighting
* Backgrounds
* Camera quality
* Object orientation
* Object appearance
* Image quality
* Differences between training and real-world data

---

## 🗂️ Project Structure

```text
AI-Waste-Classifier/
│
├── app.py
├── api.py
├── README.md
├── .gitignore
│
├── models/
│   └── waste_classifier_v2.pth
│
├── data/
│   ├── raw/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── real_world/
│
├── notebooks/
│
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── train_v2.py
│   ├── evaluate.py
│   ├── evaluate_realworld.py
│   └── predict.py
│
├── assets/
│
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

The training and evaluation datasets are excluded from the Git repository through `.gitignore`.

The trained V2 model is included:

```text
models/waste_classifier_v2.pth
```

---

## ⚙️ Installation

### 1. Prerequisites

* Windows, macOS, or Linux
* Python 3.x
* A modern web browser
* Optional: Git for cloning the repository

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd AI-Waste-Classifier
```

If you already have the project folder, simply open a terminal in the project directory.

### 3. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, the environment's Python executable can be used directly.

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision pillow pandas streamlit fastapi uvicorn python-multipart
```

---

# 🌐 Run the Modern Web Application

The main web application uses:

```text
HTML/CSS/JavaScript → FastAPI → PyTorch
```

### Step 1 — Start the FastAPI Backend

From the project root:

```bash
python -m uvicorn api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

You can also open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Step 2 — Start the Frontend

Open:

```text
frontend/index.html
```

using a local development server such as **VS Code Live Server**.

The frontend will typically be available at:

```text
http://127.0.0.1:5500/frontend/index.html
```

### Step 3 — Classify an Image

1. Open the WasteAI frontend.
2. Go to the classifier section.
3. Upload a JPG, JPEG, or PNG image.
4. Click **Analyze Waste**.
5. The frontend sends the image to the FastAPI `/predict` endpoint.
6. The PyTorch model performs inference.
7. The prediction and confidence are displayed in the browser.

---

# 🔌 API

## `GET /`

Returns basic information about the API.

Example response:

```json
{
  "message": "WasteAI API is running",
  "model": "MobileNetV3 Small",
  "classes": [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
  ]
}
```

## `POST /predict`

Accepts an uploaded image and returns the model prediction.

### Example Response

```json
{
  "success": true,
  "filename": "plastic_bottle.jpg",
  "class": "plastic",
  "confidence": 87.42,
  "disposal_guidance": "Empty and rinse plastic containers when appropriate, then place them in the appropriate plastic recycling collection.",
  "top_predictions": [
    {
      "class": "plastic",
      "confidence": 87.42
    },
    {
      "class": "glass",
      "confidence": 7.31
    },
    {
      "class": "trash",
      "confidence": 3.82
    }
  ]
}
```

---

# 🖥️ Streamlit Version

The project also contains the original Streamlit interface.

Run it with:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal.

This version provides:

* Image upload
* Waste classification
* Confidence score
* Top predictions
* Disposal guidance

The modern HTML/CSS/JavaScript frontend with FastAPI is the primary web interface.

---

# 🏋️ Training & Evaluation

The repository includes scripts for dataset preparation, model training, and evaluation.

### Prepare the Dataset

```bash
python src/prepare_dataset.py
```

### Train the Original Model

```bash
python src/train.py
```

### Train V2

```bash
python src/train_v2.py
```

### Evaluate the Model

```bash
python src/evaluate.py
```

### Evaluate Real-World Images

```bash
python src/evaluate_realworld.py
```

### Command-Line Prediction

```bash
python src/predict.py
```

Follow the instructions displayed by the script to provide an image path.

> Training and evaluation scripts require the appropriate dataset and directory structure to be available locally.

---

# 🗃️ Dataset

This project uses **TrashNet**, a waste-image dataset containing six categories:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

The original dataset contains **2,527 images**.

The project uses a 70% / 15% / 15% train-validation-test split:

| Split      |    Images |
| ---------- | --------: |
| Training   |     1,766 |
| Validation |       377 |
| Test       |       384 |
| **Total**  | **2,527** |

Dataset source:

**TrashNet — Gary Thung**

Please review the original dataset repository and its license/usage terms before redistributing the dataset.

---

# ⚠️ Limitations

* The model was trained on a relatively small, curated dataset.
* Test-set accuracy does not guarantee performance on everyday photographs.
* Some waste categories can have visually similar characteristics.
* Confidence scores represent model probabilities and are not guarantees of correctness.
* Real-world evaluation produced substantially lower accuracy than the held-out test set.
* Disposal guidance is general and may differ from local recycling regulations.
* The project is intended for learning, experimentation, and portfolio demonstration.
* It should not be used for safety-critical, regulatory, or commercial waste-sorting decisions.

---

# 🔮 Future Improvements

Possible improvements include:

* Expand the training dataset with diverse real-world images.
* Add a confidence threshold and an **"Uncertain Prediction"** state.
* Improve performance using stronger data augmentation.
* Add image-quality validation.
* Improve real-world evaluation with a larger test set.
* Add more detailed disposal instructions based on geographic location.
* Add explainable-AI visualizations such as Grad-CAM.
* Deploy the FastAPI backend and frontend online.
* Add automated API tests.
* Containerize the application with Docker.
* Add CI/CD for automated testing and deployment.

---

# 🛠️ Technologies Used

```text
Python
PyTorch
Torchvision
MobileNetV3 Small
FastAPI
Uvicorn
HTML5
CSS3
JavaScript
Streamlit
Pillow
Pandas
```

---

# 👨‍💻 Author

**Kishore N.**

BE Computer Science and Engineering
Aspiring AI/ML Engineer

---

## ⭐ Project Purpose

This project was developed as an **AI/ML portfolio project** to demonstrate practical experience with:

* Computer vision
* Image classification
* PyTorch
* Transfer learning
* Model evaluation
* REST API development
* Frontend integration
* End-to-end AI application development

Built with **PyTorch, MobileNetV3 Small, FastAPI, HTML/CSS/JavaScript, and Streamlit**.

---
