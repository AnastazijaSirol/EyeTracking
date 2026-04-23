"""
Dataset builders for TEyeD gaze-only training.

Segmentation completely removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf


DEFAULT_IMAGE_SHAPE = (96, 96, 3)
DEFAULT_JPEG_QUALITY_OPTIONS = (35, 45, 55, 65, 75, 85, 95)


# =========================================================
# LABELS
# =========================================================

def load_labels(split_root: Path | str) -> pd.DataFrame:
    split_root = Path(split_root)
    labels_path = split_root / "labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    df = pd.read_csv(labels_path)

    required = {"filename", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(
            "labels.csv must contain columns: filename,x,y "
            f"(found: {list(df.columns)})"
        )

    return df


# =========================================================
# AUGMENTATION
# =========================================================

def augment_image_photometric(image: tf.Tensor) -> tf.Tensor:
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 1.0)

    apply_gamma = tf.random.uniform([]) < 0.5
    gamma = tf.random.uniform([], 0.95, 1.10)

    image = tf.cond(
        apply_gamma,
        lambda: tf.image.adjust_gamma(image, gamma),
        lambda: image,
    )

    apply_noise = tf.random.uniform([]) < 0.6
    noise_std = tf.random.uniform([], 0.0, 0.02)
    noise = tf.random.normal(tf.shape(image), stddev=noise_std)

    image = tf.cond(
        apply_noise,
        lambda: tf.clip_by_value(image + noise, 0.0, 1.0),
        lambda: image,
    )

    return image


def random_jpeg_recompression(
    image: tf.Tensor,
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    quality_options: Sequence[int] = DEFAULT_JPEG_QUALITY_OPTIONS,
) -> tf.Tensor:
    image = tf.clip_by_value(image, 0.0, 1.0)

    image_uint8 = tf.cast(image * 255.0, tf.uint8)

    index = tf.random.uniform([], 0, len(quality_options), dtype=tf.int32)

    def encode(q):
        return tf.image.encode_jpeg(image_uint8, quality=q)

    encoded = tf.switch_case(
        index,
        branch_fns=[lambda q=q: encode(q) for q in quality_options],
    )

    decoded = tf.image.decode_jpeg(encoded, channels=image_shape[2])
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)
    decoded.set_shape(image_shape)

    return decoded


# =========================================================
# IMAGE LOADING
# =========================================================

def _decode_image(path: tf.Tensor, image_shape) -> tf.Tensor:
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=image_shape[2])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape(image_shape)
    return image


# =========================================================
# TRAIN DATASET
# =========================================================

def build_train_loader(
    dataframe: pd.DataFrame,
    root_dir: Path | str,
    batch_size: int,
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    shuffle_buffer: int = 50000,
    seed: int = 42,
) -> tf.data.Dataset:

    root_dir = Path(root_dir)

    image_paths = [
        str(root_dir / fn) for fn in dataframe["filename"].tolist()
    ]

    gazes = dataframe[["x", "y"]].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        {"image_path": image_paths, "gaze": gazes}
    )

    dataset = dataset.shuffle(shuffle_buffer, seed=seed)

    def load(sample):
        image = _decode_image(sample["image_path"], image_shape)
        gaze = tf.cast(sample["gaze"], tf.float32)
        return image, gaze

    def augment(image, gaze):
        image = random_jpeg_recompression(image, image_shape=image_shape)
        image = augment_image_photometric(image)

        # normalize
        image = (image - 0.5) * 2.0

        return image, gaze

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


# =========================================================
# EVAL DATASET
# =========================================================

def build_eval_loader(
    dataframe: pd.DataFrame,
    root_dir: Path | str,
    batch_size: int,
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    cache: bool = True,
) -> tf.data.Dataset:

    root_dir = Path(root_dir)

    image_paths = [
        str(root_dir / fn) for fn in dataframe["filename"].tolist()
    ]

    gazes = dataframe[["x", "y"]].astype("float32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        {"image_path": image_paths, "gaze": gazes}
    )

    def load(sample):
        image = _decode_image(sample["image_path"], image_shape)
        gaze = tf.cast(sample["gaze"], tf.float32)

        image = (image - 0.5) * 2.0

        return image, gaze

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if cache:
        dataset = dataset.cache()

    return dataset.prefetch(tf.data.AUTOTUNE)


# =========================================================
# UTILS
# =========================================================

def gaze_to_pixel(gaze_xy, width, height):
    x, y = float(gaze_xy[0]), float(gaze_xy[1])
    return (x + 0.5) * width, (y + 0.5) * height


def denorm_image(images: np.ndarray) -> np.ndarray:
    return np.clip((images / 2.0) + 0.5, 0.0, 1.0)


# =========================================================
# VISUALIZATION
# =========================================================

def plot_train_samples(dataset: tf.data.Dataset, count: int = 9):
    import matplotlib.pyplot as plt

    images, gazes = next(iter(dataset))

    images = denorm_image(images.numpy())
    gazes = gazes.numpy()

    n = min(count, images.shape[0])
    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)

        img = images[i]
        h, w = img.shape[:2]

        gx, gy = gazes[i]
        px, py = gaze_to_pixel((gx, gy), w, h)

        ax.imshow(img)
        ax.scatter(px, py, c="lime", s=100, marker="+")
        ax.scatter(w / 2, h / 2, c="cyan", s=40)

        ax.set_title(f"gaze: ({gx:.3f}, {gy:.3f})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()