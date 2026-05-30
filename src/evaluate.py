import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/echomind_mpl_config")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="Model doesn't support `jit_compile=True`.*")

import tensorflow as tf

from data_utils import load_image_dataset, load_trained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MRI classifier.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a test directory containing Normal and Sick subfolders.",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "final_best_model.keras"),
        help="Path to the trained Keras model.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        model = load_trained_model(args.model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    dataset = load_image_dataset(args.data, batch_size=args.batch_size, shuffle=False)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        loss, accuracy, precision, recall = model.evaluate(dataset, verbose=1)

    print("\nEvaluation results")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Loss:      {loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
