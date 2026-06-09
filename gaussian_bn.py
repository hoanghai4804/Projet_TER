# gaussian_bn.py
"""
Test Gaussian Bayesian Network sur le dataset réel Aissou.
Contrairement au modèle discret, le GBN traite directement
les valeurs continues sans discrétisation.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import os


def load_continuous_dataset(
        path="data/GPS_Data_Simplified_2D_Feature_Map.xlsx",
        n_samples=2000):
    """
    Charge le dataset réel SANS discrétisation.
    Garde les valeurs continues pour le Gaussian BN.
    """

    print("Chargement du dataset (valeurs continues)...")
    df = pd.read_excel(path)

    # Label binaire
    df["S"] = (df["Output"] != 0).astype(int)

    # Features continues (les 4 mêmes qu'avant, mais NON discrétisées)
    features = ["CN0", "DO", "PQP", "PD"]

    # Dataset équilibré
    n_per = n_samples // 2
    df_normal = df[df["S"]==0].sample(n=n_per, random_state=42)
    df_attack = df[df["S"]==1].sample(n=n_per, random_state=42)

    df_bal = pd.concat([df_normal, df_attack])
    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_bal[features].values
    y = df_bal["S"].values

    print(f" Dataset chargé : {len(df_bal)} échantillons")
    print(f"   Features (continues) : {features}")

    return X, y, features


def gaussian_bayes_test():
    """
    Gaussian Naive Bayes = Gaussian Bayesian Network
    avec structure Naïve Bayes (S → features).

    Chaque P(feature | S) est modélisée par une
    distribution gaussienne au lieu d'une table discrète.
    """

    # ── Charger données continues ──────────────
    X, y, features = load_continuous_dataset()

    # ── Standardisation (important pour Gaussian) ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train/test split ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # ── Gaussian Naive Bayes ──────────────────
    # Équivalent à un Gaussian Bayesian Network
    # avec structure Naïve Bayes
    print("\nEntraînement Gaussian Bayesian Network...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    # ── Prédiction ────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(S=1)

    # ── Évaluation ────────────────────────────
    print("\n=== Résultats Gaussian BN sur données réelles ===\n")

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    detection_rate = TP / (TP + FN)
    false_alarm    = FP / (FP + TN)
    precision      = TP / (TP + FP)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy           : {acc*100:.2f}%")
    print(f"Taux de détection  : {detection_rate*100:.2f}%")
    print(f"Taux fausse alarme : {false_alarm*100:.2f}%")
    print(f"Précision          : {precision*100:.2f}%")
    print(f"AUC                : {roc_auc:.4f}")

    print(f"\nMatrice de confusion :")
    print(f"   TN={TN}  FP={FP}")
    print(f"   FN={FN}  TP={TP}")

    print("\nRapport de classification :")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal", "Attaque"]
    ))

    # ── Afficher les paramètres gaussiens appris ──
    print("=== Paramètres gaussiens appris ===\n")
    print(f"{'Feature':<8} {'Mean Normal':>12} {'Mean Attack':>12} "
          f"{'Var Normal':>12} {'Var Attack':>12}")
    print("-" * 60)
    for i, feat in enumerate(features):
        mean_n = model.theta_[0][i]
        mean_a = model.theta_[1][i]
        var_n  = model.var_[0][i]
        var_a  = model.var_[1][i]
        print(f"{feat:<8} {mean_n:>12.3f} {mean_a:>12.3f} "
              f"{var_n:>12.3f} {var_a:>12.3f}")

    # ── ROC Curve ─────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="seagreen", linewidth=2,
             label=f"Gaussian BN (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray",
             linestyle="--", label="Aléatoire (AUC = 0.5)")
    plt.xlabel("Taux de fausse alarme (FPR)")
    plt.ylabel("Taux de détection (TPR)")
    plt.title("Courbe ROC — Gaussian Bayesian Network\n"
              "sur dataset réel (Aissou 2022)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/roc_gaussian.png", dpi=150)
    plt.close()
    print("\n ROC Gaussian sauvegardée : results/roc_gaussian.png")

    return {
        "accuracy": acc,
        "detection_rate": detection_rate,
        "false_alarm": false_alarm,
        "precision": precision,
        "auc": roc_auc
    }


def compare_all(results_gaussian):
    """
    Comparaison finale : Discrete (réel) vs Gaussian (réel)
    """

    # Résultats discrete réel (du test précédent)
    discrete_real = {
        "accuracy": 0.6040,
        "detection_rate": 0.6560,
        "false_alarm": 0.4480,
        "precision": 0.5942,
        "auc": 0.6212
    }

    print("\n" + "=" * 55)
    print("  COMPARAISON — Discrete vs Gaussian (données réelles)")
    print("=" * 55)
    print(f"{'Métrique':<22} {'Discrete BN':>14} {'Gaussian BN':>14}")
    print("-" * 55)

    metrics = [
        ("Accuracy",           "accuracy"),
        ("Taux de détection",  "detection_rate"),
        ("Taux fausse alarme", "false_alarm"),
        ("Précision",          "precision"),
        ("AUC",                "auc"),
    ]
    for label, key in metrics:
        vd = discrete_real[key]     * 100
        vg = results_gaussian[key]  * 100
        diff = vg - vd
        print(f"{label:<22} {vd:>13.2f}% {vg:>13.2f}%  "
              f"({diff:+.1f})")

    print("=" * 55)


if __name__ == "__main__":

    print("=" * 55)
    print("  TEST GAUSSIAN BAYESIAN NETWORK")
    print("=" * 55)

    results = gaussian_bayes_test()
    compare_all(results)

    print("\n Test terminé !")