import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/echomind_mpl_config")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="Model doesn't support `jit_compile=True`.*")

from data_utils import load_trained_model, prediction_label, prepare_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Normal or Sick for one MRI image.")
    parser.add_argument("image", help="Path to an MRI image.")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "final_best_model.keras"),
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for the Sick class.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image = prepare_image(args.image)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        model = load_trained_model(args.model)
        score = float(model.predict(image, verbose=0)[0][0])

    label = prediction_label(score, args.threshold)

    print(f"Prediction: {label}")
    print(f"Sick probability: {score:.4f}")
    print(f"Threshold: {args.threshold:.2f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
