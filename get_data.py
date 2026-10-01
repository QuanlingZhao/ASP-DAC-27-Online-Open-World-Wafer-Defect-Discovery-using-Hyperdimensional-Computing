import os
import sys
import json
import time
import argparse
import random
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -----------------------------
# Labels
# -----------------------------
DEFECT_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch"
]
ALL_CLASSES = DEFECT_CLASSES + ["none"]

NONE_ALIASES = {
    "none", "normal", "non-defect", "nondefect",
    "no defect", "no-defect"
}


def normalize_failure_label(x) -> str:
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]

    s = str(x).strip()
    s_low = s.lower()

    exact_map = {
        "center": "Center",
        "donut": "Donut",
        "edge-loc": "Edge-Loc",
        "edge loc": "Edge-Loc",
        "edge-ring": "Edge-Ring",
        "edge ring": "Edge-Ring",
        "loc": "Loc",
        "local": "Loc",
        "near-full": "Near-full",
        "near full": "Near-full",
        "random": "Random",
        "scratch": "Scratch",
        "none": "none",
        "normal": "none",
        "non-defect": "none",
        "nondefect": "none",
        "no defect": "none",
        "no-defect": "none",
    }

    if s_low in exact_map:
        return exact_map[s_low]

    for k, v in exact_map.items():
        if k in s_low:
            return v

    return s


# -----------------------------
# Load old WM-811K pickle
# -----------------------------
def load_wm811k_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        sys.modules["pandas.indexes"] = pd.core.indexes
        try:
            df = pd.read_pickle(path)
        except Exception as e1:
            print(f"[load] pd.read_pickle failed: {e1}")
            with open(path, "rb") as f:
                df = pickle.load(f, encoding="latin1")
            print("[load] loaded with pickle.load(..., encoding='latin1')")
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    if isinstance(df, dict):
        df = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Loaded object is not a DataFrame, got {type(df)}")

    print("Loaded columns:", list(df.columns))

    if "waferMap" not in df.columns:
        raise ValueError("Expected column 'waferMap'")
    if "failureType" not in df.columns:
        raise ValueError("Expected column 'failureType'")

    labels = df["failureType"].apply(normalize_failure_label)
    keep = labels.isin(ALL_CLASSES)

    out = pd.DataFrame({
        "waferMap": df.loc[keep, "waferMap"].values,
        "label": labels.loc[keep].values,
    }).reset_index(drop=True)

    return out


# -----------------------------
# Image conversion
# -----------------------------
def wafer_to_rgb_image(arr: np.ndarray, out_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D waferMap, got shape={a.shape}")

    rgb = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    rgb[a == 0] = [0, 0, 0]
    rgb[a == 1] = [0, 255, 255]
    rgb[a == 2] = [255, 0, 255]

    img = Image.fromarray(rgb)
    img = img.resize(out_size, Image.BILINEAR)
    return img


def rotate_wafer_map(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, k=k)


def augment_train_minority_classes(
    train_df: pd.DataFrame,
    target_per_class: int = 2500,
    aug_classes: Tuple[str, ...] = ("Donut", "Near-full", "Random", "Scratch"),
    none_cap: int = 12000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = [{"waferMap": r["waferMap"], "label": r["label"]} for _, r in train_df.iterrows()]

    for cls in aug_classes:
        sub = train_df[train_df["label"] == cls]
        n = len(sub)
        if n == 0 or n >= target_per_class:
            continue

        need = target_per_class - n
        sampled_idx = rng.choice(sub.index.to_numpy(), size=need, replace=True)

        for idx in sampled_idx:
            row = sub.loc[idx]
            k = int(rng.choice([1, 2, 3]))
            aug_map = rotate_wafer_map(np.asarray(row["waferMap"]), k)
            rows.append({"waferMap": aug_map, "label": cls})

    out = pd.DataFrame(rows)

    if none_cap is not None and none_cap > 0:
        none_df = out[out["label"] == "none"]
        other_df = out[out["label"] != "none"]
        if len(none_df) > none_cap:
            none_df = none_df.sample(n=none_cap, random_state=seed)
        out = pd.concat([other_df, none_df], ignore_index=True)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# -----------------------------
# Dataset
# -----------------------------
class WaferDataset(Dataset):
    def __init__(
        self,
        wafer_maps: List[np.ndarray],
        labels: List[str],
        label_encoder: LabelEncoder = None,
        fit_label_encoder: bool = False,
        augment: bool = False,
        image_size: int = 224,
    ):
        self.wafer_maps = list(wafer_maps)
        self.labels_str = list(labels)

        if fit_label_encoder:
            self.le = LabelEncoder()
            self.le.fit(self.labels_str)
        else:
            if label_encoder is None:
                raise ValueError("label_encoder required when fit_label_encoder=False")
            self.le = label_encoder

        self.labels = self.le.transform(self.labels_str)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if augment:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.wafer_maps)

    def __getitem__(self, idx: int):
        img = wafer_to_rgb_image(self.wafer_maps[idx], out_size=(self.image_size, self.image_size))
        x = self.tf(img)
        y = int(self.labels[idx])
        return x, y



def get_data(wm811k_pkl,test_size,val_size,seed,none_cap,image_size,batch_size,num_workers):
    print("Loading WM-811K dataframe...")
    df = load_wm811k_df(wm811k_pkl)
    print("\nRaw counts:")
    print(df["label"].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"],
    )


    train_df_aug = augment_train_minority_classes(
        train_df,
        target_per_class=2500,
        aug_classes=("Donut", "Near-full", "Random", "Scratch"),
        none_cap=none_cap,
        seed=seed,
    )

    print("\nTrain counts before augmentation:")
    print(train_df["label"].value_counts())
    print("\nTrain counts after augmentation/cap:")
    print(train_df_aug["label"].value_counts())
    print("\nVal counts:")
    print(val_df["label"].value_counts())
    print("\nTest counts:")
    print(test_df["label"].value_counts())

    train_ds = WaferDataset(
        train_df_aug["waferMap"].tolist(),
        train_df_aug["label"].tolist(),
        fit_label_encoder=True,
        augment=True,
        image_size=image_size,
    )
    le = train_ds.le

    val_ds = WaferDataset(
        val_df["waferMap"].tolist(),
        val_df["label"].tolist(),
        label_encoder=le,
        fit_label_encoder=False,
        augment=False,
        image_size=image_size,
    )
    test_ds = WaferDataset(
        test_df["waferMap"].tolist(),
        test_df["label"].tolist(),
        label_encoder=le,
        fit_label_encoder=False,
        augment=False,
        image_size=image_size,
    )

    print("\nClass order:")
    print(list(le.classes_))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, le






if __name__ == "__main__":
   train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, le = get_data("../data/LSWMD.pkl",0.2,0.1,42,12000,224,32,4)



