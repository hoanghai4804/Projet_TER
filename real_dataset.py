# real_dataset.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve


def find_optimal_threshold(y_true, scores):
    """Seuil optimal selon l'indice de Youden : J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def load_real_dataset(
        path="data/GPS_Data_Simplified_2D_Feature_Map.xlsx",
        n_samples=2000):

    print("Chargement du dataset réel...")
    df = pd.read_excel(path)
    print(f"Dataset chargé : {len(df)} lignes, "
          f"{len(df.columns)} colonnes")

    # ── Label binaire ──────────────────────────
    df["S"] = (df["Output"] != 0).astype(int)
    print(f"\nDistribution:")
    print(f"  Normal  (S=0) : {(df['S']==0).sum()}")
    print(f"  Attaque (S=1) : {(df['S']==1).sum()}")

    # ── Thresholds optimaux (Youden) ───────────
    df_s = df.sample(n=20000, random_state=42)
    y_s  = df_s["S"].values

    print("\nCalcul des thresholds optimaux...")

    thresh_DO  = find_optimal_threshold(y_s, df_s["DO"].abs())
    thresh_PD  = find_optimal_threshold(y_s, df_s["PD"].abs())
    thresh_CN0 = find_optimal_threshold(y_s, -df_s["CN0"])
    thresh_PQP = find_optimal_threshold(y_s, df_s["PQP"].abs())

    print(f"  DO  (Doppler)     abs > {thresh_DO:.2f}  -> D=1")
    print(f"  PD  (Pseudorange) abs > {thresh_PD:.2f}  -> P=1")
    print(f"  CN0 (C/N0)            < {-thresh_CN0:.2f} -> C=1")
    print(f"  PQP (Quality)     abs > {thresh_PQP:.2f}  -> A=1")

    # ── Discrétisation ─────────────────────────
    df["C"] = (df["CN0"]      < -thresh_CN0  ).astype(int)
    df["D"] = (df["DO"].abs() > thresh_DO    ).astype(int)
    df["A"] = (df["PQP"].abs()> thresh_PQP   ).astype(int)
    df["P"] = (df["PD"].abs() > thresh_PD    ).astype(int)

    # ── Vérification ───────────────────────────
    print(f"\n=== Taux d'anomalies après discrétisation ===")
    df_n = df[df["S"] == 0]
    df_a = df[df["S"] == 1]

    for col in ["C", "D", "A", "P"]:
        rate_n = df_n[col].mean() * 100
        rate_a = df_a[col].mean() * 100
        diff   = rate_a - rate_n
        ok     = "OK" if diff > 5 else "WARN"
        print(f"  {col} -> normal={rate_n:.1f}%  "
              f"attaque={rate_a:.1f}%  diff={diff:+.1f}% {ok}")

    # ── Dataset équilibré ──────────────────────
    n_per = min(n_samples // 2,
                (df["S"]==0).sum(),
                (df["S"]==1).sum())

    df_final = pd.concat([
        df[df["S"]==0].sample(n=n_per, random_state=42),
        df[df["S"]==1].sample(n=n_per, random_state=42)
    ])
    df_final = df_final.sample(
        frac=1, random_state=42).reset_index(drop=True)
    df_final = df_final[["S", "C", "D", "A", "P"]]

    print(f"\nDataset reel pret : {len(df_final)} echantillons")
    return df_final


def compare_datasets(df_sim, df_real):
    """Tableau comparatif simulé vs réel."""

    print("\n=== Comparaison Simulé vs Réel ===\n")
    print(f"{'Variable':<10} {'Sim Normal':>11} {'Réel Normal':>12} "
          f"{'Sim Attack':>11} {'Réel Attack':>12}")
    print("-" * 60)

    for col in ["C", "D", "A", "P"]:
        sn = df_sim [df_sim ["S"]==0][col].mean()*100
        sa = df_sim [df_sim ["S"]==1][col].mean()*100
        rn = df_real[df_real["S"]==0][col].mean()*100
        ra = df_real[df_real["S"]==1][col].mean()*100
        print(f"{col:<10} {sn:>10.1f}% {rn:>11.1f}% "
              f"{sa:>10.1f}% {ra:>11.1f}%")


if __name__ == "__main__":
    df = load_real_dataset()
    print(df.head())