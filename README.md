# Face Access Control System

A two-stage deep learning pipeline that first determines whether a human face is visible (not covered by a mask, hoodie, or any other obstruction), and then — only if the face is clearly visible — classifies the individual as **Authorized** or **Unauthorized**.

Built with TensorFlow/Keras (MobileNetV2) for the visibility stage and DeepFace (VGG-Face) for the authorization stage. Deployed as an interactive Streamlit web application.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Stage 1 — Face Visibility Classification](#stage-1--face-visibility-classification)
4. [Stage 2 — Face Authorization](#stage-2--face-authorization)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Running the Streamlit App](#running-the-streamlit-app)
8. [Using the App](#using-the-app)
9. [Training Details](#training-details)
10. [Model Files](#model-files)
11. [Dependencies](#dependencies)

---

## Project Overview

Many access control systems fail when individuals attempt to bypass identification by covering their faces with masks, hoodies, scarves, or other obstructions. This project solves that with a **gated two-stage pipeline**:

- **Gate (Stage 1):** Is this face covered or uncovered?
  - If **covered** → reject immediately. No identification is attempted.
  - If **uncovered** → proceed to Stage 2.
- **Classification (Stage 2):** Is this person authorized or unauthorized?
  - Compare the uncovered face against a registered database of authorized individuals.
  - Return **Authorized** (with the matched person's name) or **Unauthorized**.

This design prevents a common spoofing vector: a covered face can never be "authorized" regardless of how similar its partial features are to a registered person.

---

## Pipeline Architecture

```
Input Image
     │
     ▼
┌────────────────────────────┐
│   Stage 1: Visibility      │
│   MobileNetV2 (fine-tuned) │
│   Covered / Uncovered      │
└────────────┬───────────────┘
             │
      Covered?──────────────► STOP: "Face is covered"
             │
          Uncovered
             │
             ▼
┌────────────────────────────┐
│   Stage 2: Authorization   │
│   DeepFace — VGG-Face      │
│   Embedding distance vs.   │
│   authorized_faces/        │
└────────────┬───────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
Authorized       Unauthorized
(name shown)
```

---

## Stage 1 — Face Visibility Classification

### Goal
Detect whether a human face is fully visible or partially/fully obstructed by anything — a mask, hoodie, scarf, clothing, or any other covering.

### Model
- **Architecture:** MobileNetV2 pre-trained on ImageNet, with a custom classification head
- **Custom head:** `AveragePooling2D(7×7) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(2, Softmax)`
- **Input size:** 224 × 224 × 3
- **Output classes:**
  - `0` — WithMask (Covered)
  - `1` — WithoutMask (Uncovered)

### Dataset
[Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) downloaded via `kagglehub`.

| Split      | Images |
|------------|--------|
| Train      | 10,000 |
| Validation | 800    |

The dataset contains real-world photos of people both with and without face masks. The model learns general occlusion cues, so it generalises to other types of face coverings beyond just masks.

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model     | MobileNetV2 (layers frozen) |
| Optimizer      | Adam (lr = 1e-4) |
| Loss function  | Categorical cross-entropy |
| Epochs         | 2 |
| Batch size     | 32 |
| Augmentation   | Rotation ±20°, horizontal flip |

### Results

| Epoch | Train Accuracy | Validation Accuracy |
|-------|---------------|---------------------|
| 1     | 92.96%        | 98.37%              |
| 2     | 97.89%        | **99.00%**          |

The MobileNetV2 base is kept frozen — only the custom head is trained. This achieves 99% validation accuracy in just 2 epochs by leveraging ImageNet transfer learning.

### Inference Preprocessing
At inference time, every image is:
1. Converted to RGB
2. Resized to 224 × 224
3. Pixel values normalized to `[0, 1]` (divided by 255)

---

## Stage 2 — Face Authorization

### Goal
Verify whether an uncovered face matches any individual in the authorized-persons database.

### Approach
Stage 2 uses **DeepFace** with the **VGG-Face** backbone — a pre-trained face embedding model that maps any face to a high-dimensional feature vector. No retraining is required to add or remove authorized individuals.

**How verification works:**
1. The probe face is encoded into a VGG-Face embedding vector.
2. That vector is compared (cosine distance) against the embedding of every image in `authorized_faces/`.
3. The **best match** (lowest distance) is selected.
4. If best distance ≤ VGG-Face threshold (0.40) → **Authorized**.
5. Otherwise → **Unauthorized**.

### Confidence Score
```
confidence = 1.0 - (best_distance / threshold)
```
- 100% confidence = distance of 0 (identical embedding)
- 0% confidence = distance exactly at the threshold boundary

### Authorized Faces Database
Authorized individuals are managed through a simple folder — no retraining or code changes needed:

```
authorized_faces/
├── alice.jpg       ← registered as "alice"
├── bob.png         ← registered as "bob"
└── carol.jpeg      ← registered as "carol"
```

- The **filename without extension** becomes the person's display name shown in results.
- Multiple reference photos per person can be registered using numbered filenames (`alice_1.jpg`, `alice_2.jpg`).
- Photos can be added or deleted at any time through the app sidebar.

---

## Project Structure

```
final_project/
├── Group8.ipynb          # Full training + pipeline notebook
├── my_model.keras        # Trained visibility model (native Keras format)
├── visibility_model.h5   # Same model in legacy HDF5 format
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── authorized_faces/     # Auto-created — add reference photos here
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Anaconda (recommended) or any virtual environment

### 1. Navigate to the project folder
```bash
cd machine_learning_workshop/final_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **First-run note:** DeepFace will automatically download the VGG-Face model weights (~500 MB) the first time it runs. Make sure you have an internet connection.

### 3. Confirm the model file is present
```
my_model.keras   ← must be in the same directory as app.py
```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## Using the App

### Step 1 — Register Authorized People (Sidebar)

1. Open the **left sidebar** and find the **"Registered Faces"** section.
2. Click **"Browse files"** and select clear, front-facing photos of authorized individuals.
3. **Name each file after the person before uploading** — e.g., `john_doe.jpg`. The filename becomes the identity label shown in results.
4. Uploaded photos are saved to `authorized_faces/` and are immediately active.
5. To remove someone, click the **✕** button next to their name.

> **Tips for best accuracy:**
> - Use a well-lit, front-facing photo with no obstruction.
> - Avoid sunglasses, heavy shadows, or extreme angles in reference photos.
> - Register multiple photos of the same person for more robust matching.

### Step 2 — Test a Face Image (Main Area)

1. Click **"Upload a face image to run through the pipeline"**.
2. Upload any `.jpg`, `.jpeg`, or `.png` containing a human face.
3. Results appear automatically:

| Scenario | Result |
|----------|--------|
| Face is covered | Red banner — "Stage 1: Covered". Pipeline stops. |
| Face uncovered, person authorized | Two green banners. Matched person's name shown. |
| Face uncovered, person unauthorized | Green Stage 1 banner, red Stage 2 banner. |
| No authorized faces registered yet | Green Stage 1 banner, yellow warning for Stage 2. |

### Step 3 — Download the Model (Optional)

Click **"Download visibility model"** in the sidebar to save `my_model.keras` locally.

---

## Training Details

The notebook `Group8.ipynb` documents the full process cell by cell:

| Section | Cells | Description |
|---------|-------|-------------|
| Setup | 1–2 | Library imports (TensorFlow, Keras, PIL, kagglehub) |
| Data | 3–4 | Download Face Mask dataset, visualise sample images |
| Stage 1 Training | 5–7 | Build MobileNetV2 model, train 2 epochs, save weights |
| Stage 2 Setup | 8–11 | Install DeepFace, load visibility model, define pipeline functions |
| Demo | 12–13 | Instructions and live demo cell — swap in any test image path |
| Export | 14–15 | Verify `app.py`, `requirements.txt`, and `my_model.keras` exist |

---

## Model Files

| File | Format | Notes |
|------|--------|-------|
| `my_model.keras` | Native Keras | Recommended — use with TensorFlow 2.12+ |
| `visibility_model.h5` | HDF5 (legacy) | Kept for backwards compatibility |

Both files contain identical trained weights.

**Load and run the visibility model manually:**
```python
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("my_model.keras")

img = Image.open("face.jpg").convert("RGB").resize((224, 224))
arr = np.array(img, dtype=np.float32) / 255.0
pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]

# 0 = Covered, 1 = Uncovered (alphabetical class order from training)
label = "Uncovered" if np.argmax(pred) == 1 else "Covered"
print(f"{label} ({float(np.max(pred)):.2%} confidence)")
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.12.0 | Stage 1 model training and inference |
| `streamlit` | ≥ 1.30.0 | Web application framework |
| `deepface` | ≥ 0.0.93 | Stage 2 face verification (VGG-Face) |
| `Pillow` | ≥ 9.0.0 | Image loading and preprocessing |
| `numpy` | ≥ 1.23.0 | Array operations |
| `opencv-python-headless` | ≥ 4.8.0 | Image decoding used internally by DeepFace |

Install everything:
```bash
pip install -r requirements.txt
```

---

## Authors
- [Obafisayo Abimbola](https://github.com/obafisayo)
- [Tomison](https://github.com/)
- [Warris](https://github.com/)
- [Abiola](https://github.com/)

**Group 8** — Machine Learning & Deep Learning Workshop Program Final Project
