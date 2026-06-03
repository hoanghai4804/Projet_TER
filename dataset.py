# dataset.py
import pandas as pd
import numpy as np
from pgmpy.sampling import BayesianModelSampling
from model import build_model


def generate_dataset(n_per_class=1000, random_seed=42):
    """
    Génère un dataset équilibré en utilisant
    BayesianModelSampling de pgmpy.

    n_per_class → nombre d'échantillons par classe
    (1000 normaux + 1000 attaques = 2000 total)
    """

    np.random.seed(random_seed)
    model, _ = build_model()
    sampler = BayesianModelSampling(model)

    # ── Sinh mẫu BÌNH THƯỜNG (S=0) ────────────
    # rejected_sampling giữ chỉ những mẫu có S=0
    print("Génération des échantillons normaux (S=0)...")
    df_normal = sampler.rejection_sample(
        evidence=[("S", 0)],
        size=n_per_class,
        show_progress=False
    )

    # ── Sinh mẫu BỊ TẤN CÔNG (S=1) ───────────
    print("Génération des échantillons attaqués (S=1)...")
    df_attack = sampler.rejection_sample(
        evidence=[("S", 1)],
        size=n_per_class,
        show_progress=False
    )

    # ── Ghép và trộn ──────────────────────────
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Đảm bảo kiểu dữ liệu là int
    for col in ["S", "C", "D", "A", "P"]:
        df[col] = df[col].astype(int)

    return df


def save_dataset(df, path="data/dataset.csv"):
    """Lưu dataset ra file CSV"""
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Dataset sauvegardé : {path}")


def describe_dataset(df):
    """In thống kê mô tả"""

    print(f"\n=== Statistiques du dataset ===")
    print(f"Total          : {len(df)} échantillons")
    print(f"Normal  (S=0)  : {(df['S']==0).sum()}")
    print(f"Attaque (S=1)  : {(df['S']==1).sum()}")

    print(f"\n=== Taux d'anomalies par variable ===")
    for col in ["C", "D", "A", "P"]:
        rate_normal = df[df["S"]==0][col].mean() * 100
        rate_attack = df[df["S"]==1][col].mean() * 100
        print(f"{col} → normal={rate_normal:.1f}%  attaque={rate_attack:.1f}%")

    print(f"\n=== Valeurs CPT attendues ===")
    print(f"C → normal=5.0%   attaque=90.0%")
    print(f"D → normal=3.0%   attaque=70.0%")
    print(f"A → normal=2.0%   attaque=40.0%")
    print(f"P → normal=4.0%   attaque=65.0%")


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    describe_dataset(df)