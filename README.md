# EchoMind

EchoMind is a cardiac MRI binary classification project that predicts whether an MRI image is `Normal` or `Sick`.

The model uses MobileNetV2 transfer learning, followed by fine-tuning on the final layers. The full training workflow is documented in `notebooks/classification.ipynb`, and the trained model is included in `models/final_best_model.keras`.

## Project Structure

```text
echomind/
├── app/
│   ├── main.py
│   ├── static/
│   └── templates/
├── notebooks/
│   └── classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── final_best_model.keras
├── reports/
│   ├── metrics.md
│   └── mri_features_analysis.csv
├── requirements.txt
└── .gitignore
```

## Model

- Base architecture: MobileNetV2
- Pretraining: ImageNet
- Input size: `224 x 224 x 3`
- Classifier head:
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - Dropout(0.5)
  - Dense(1, Sigmoid)
- Classes:
  - `Normal`
  - `Sick`

## Results

| Dataset | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: |
| Internal test set | 0.9726 | 0.9935 | 0.9390 |
| External ACDC set | 0.9000 | 0.9667 | 0.8286 |

More details are available in `reports/metrics.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Web App

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The app lets you upload one MRI image and returns the predicted class with Normal/Sick probabilities.

## Predict One Image

```bash
python src/predict.py path/to/image.png
```

Example output:

```text
Prediction: Sick
Sick probability: 0.8732
Threshold: 0.50
```

## Evaluate a Test Folder

The evaluation folder should contain two subfolders named `Normal` and `Sick`.

```text
test_data/
├── Normal/
└── Sick/
```

Run:

```bash
python src/evaluate.py --data path/to/test_data
```

## Dataset Notice

The full MRI datasets are not included in this repository because of size and licensing considerations. The repository includes the trained model, reproducible notebook, evaluation scripts, and reported metrics.
