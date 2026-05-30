import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/echomind_mpl_config")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Sick"]


def load_trained_model(model_path, compile_model=False):
    """Load the saved Keras model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=compile_model)


def prepare_image(image_path, image_size=IMAGE_SIZE):
    """Load one image and apply the same preprocessing used during training."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def prepare_image_bytes(image_bytes, image_size=IMAGE_SIZE):
    """Prepare an uploaded image without saving it to disk."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(image_size, Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def normalize_batch(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.cast(labels, tf.float32)
    return images, labels


def load_image_dataset(dataset_dir, batch_size=32, shuffle=False):
    """Load a Normal/Sick directory dataset for evaluation."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        interpolation="nearest",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataset.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE
    )


def prediction_label(score, threshold=0.5):
    label_index = int(score >= threshold)
    return CLASS_NAMES[label_index]
