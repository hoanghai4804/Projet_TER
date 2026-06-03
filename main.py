# main.py
"""
Projet TER — Détection d'attaques GPS Spoofing pour UAV
            via réseaux bayésiens

Pipeline complet :
  1. Construction du modèle bayésien
  2. Génération du dataset simulé
  3. Inférence et évaluation
  4. Visualisations
"""

from model import build_model
from dataset import generate_dataset, save_dataset, describe_dataset
from evaluation import predict_proba, evaluate, save_results
from visualisation import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_probability_distribution,
    plot_anomaly_rates
)


def main():

    print("=" * 55)
    print("  PROJET TER — Détection GPS Spoofing (UAV)")
    print("  Réseau Bayésien Naïf")
    print("=" * 55)

    # ─────────────────────────────────────────
    # ÉTAPE 1 — Construction du modèle
    # ─────────────────────────────────────────
    print("\n[1/4] Construction du modèle bayésien...")
    model, inference = build_model()

    # ─────────────────────────────────────────
    # ÉTAPE 2 — Génération du dataset
    # ─────────────────────────────────────────
    print("\n[2/4] Génération du dataset simulé...")
    df = generate_dataset(n_per_class=1000)
    save_dataset(df)
    describe_dataset(df)

    # ─────────────────────────────────────────
    # ÉTAPE 3 — Inférence et évaluation
    # ─────────────────────────────────────────
    print("\n[3/4] Inférence et évaluation...")
    y_proba = predict_proba(inference, df)
    y_pred  = (y_proba > 0.5).astype(int)
    y_true  = df["S"].values

    results = evaluate(y_true, y_pred, y_proba)
    save_results(results)

    # ─────────────────────────────────────────
    # ÉTAPE 4 — Visualisations
    # ─────────────────────────────────────────
    print("\n[4/4] Génération des visualisations...")
    plot_roc_curve(results)
    plot_confusion_matrix(results)
    plot_probability_distribution(results)
    plot_anomaly_rates(df)

    print("\n" + "=" * 55)
    print("  ✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("  Résultats disponibles dans : results/")
    print("=" * 55)


if __name__ == "__main__":
    main()