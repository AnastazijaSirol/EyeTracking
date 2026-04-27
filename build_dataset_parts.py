"""
Dataset builders for TEyeD parts-only training (segmentation only).

Gaze completely removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================================================
# CONFIG
# =========================================================

SEG_PARTS = ("pupil", "iris", "lid")

DEFAULT_IMAGE_SHAPE = (96, 96, 3)
DEFAULT_MASK_SHAPE = (96, 96, 1)
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

    if "filename" not in df.columns:
        raise ValueError("labels.csv must contain 'filename' column")

    return df


# =========================================================
# AUGMENTATION
# =========================================================

def augment_image_photometric(image: tf.Tensor) -> tf.Tensor:
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, 0.9, 1.1)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def random_jpeg_recompression(
    image: tf.Tensor,
    *,
    image_shape=DEFAULT_IMAGE_SHAPE,
    quality_options=DEFAULT_JPEG_QUALITY_OPTIONS,
):
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
# LOADERS
# =========================================================

def _decode_image(path, image_shape):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=image_shape[2])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape(image_shape)
    return img


def _decode_mask(path, mask_shape):
    mask = tf.image.decode_png(tf.io.read_file(path), channels=mask_shape[2])
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask.set_shape(mask_shape)
    return mask


def _get_mask_paths(df, root_dir, seg_parts):
    root_dir = Path(root_dir)

    mask_paths = {
        part: [
            str(root_dir / f"seg_{part}_2D" / Path(fn).with_suffix(".png"))
            for fn in df["filename"].tolist()
        ]
        for part in seg_parts
    }

    return mask_paths


# =========================================================
# TRAIN DATASET
# =========================================================

def build_train_loader(
    dataframe,
    root_dir,
    batch_size,
    *,
    image_shape=DEFAULT_IMAGE_SHAPE,
    mask_shape=DEFAULT_MASK_SHAPE,
    seg_parts=SEG_PARTS,
    shuffle_buffer=50000,
):

    root_dir = Path(root_dir)

    image_paths = [
        str(root_dir / fn) for fn in dataframe["filename"].tolist()
    ]

    mask_paths = _get_mask_paths(dataframe, root_dir, seg_parts)

    slices = {"image_path": image_paths}
    for part in seg_parts:
        slices[f"mask_{part}"] = mask_paths[part]

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.shuffle(shuffle_buffer)

    def load(sample):
        image = _decode_image(sample["image_path"], image_shape)

        masks = [
            _decode_mask(sample[f"mask_{p}"], mask_shape)
            for p in seg_parts
        ]

        mask = tf.concat(masks, axis=-1)  # stacked

        return image, mask

    def augment(image, mask):
        image = random_jpeg_recompression(image, image_shape=image_shape)
        image = augment_image_photometric(image)

        image = (image - 0.5) * 2.0

        return image, mask

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


# =========================================================
# EVAL DATASET
# =========================================================

def build_eval_loader(
    dataframe,
    root_dir,
    batch_size,
    *,
    image_shape=DEFAULT_IMAGE_SHAPE,
    mask_shape=DEFAULT_MASK_SHAPE,
    seg_parts=SEG_PARTS,
    cache=True,
):

    root_dir = Path(root_dir)

    image_paths = [
        str(root_dir / fn) for fn in dataframe["filename"].tolist()
    ]

    mask_paths = _get_mask_paths(dataframe, root_dir, seg_parts)

    slices = {"image_path": image_paths}
    for part in seg_parts:
        slices[f"mask_{part}"] = mask_paths[part]

    ds = tf.data.Dataset.from_tensor_slices(slices)

    def load(sample):
        image = _decode_image(sample["image_path"], image_shape)

        masks = [
            _decode_mask(sample[f"mask_{p}"], mask_shape)
            for p in seg_parts
        ]

        mask = tf.concat(masks, axis=-1)

        image = (image - 0.5) * 2.0

        return image, mask

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if cache:
        ds = ds.cache()

    return ds.prefetch(tf.data.AUTOTUNE)


# =========================================================
# VISUALIZATION
# =========================================================

def plot_train_samples(dataset, count=9):
    import matplotlib.pyplot as plt

    images, masks = next(iter(dataset))

    images = (images.numpy() / 2.0) + 0.5
    masks = masks.numpy()

    n = min(count, images.shape[0])

    plt.figure(figsize=(15, 15))

    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])

        for c in range(masks.shape[-1]):
            plt.imshow(masks[i, :, :, c], alpha=0.3)

        plt.title(f"Sample {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
