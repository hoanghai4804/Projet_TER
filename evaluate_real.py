# evaluate_real.py
from model import build_model
from real_dataset import load_real_dataset, compare_datasets
from dataset import generate_dataset
from evaluation import predict_proba, evaluate
from visualisation import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_probability_distribution
)
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_comparison(results_sim, results_real,
                    save_path="results/comparison.png"):

    metrics = ["accuracy", "detection_rate", "false_alarm", "auc"]
    labels  = ["Accuracy", "Taux détection",
               "Taux fausse alarme", "AUC"]

    sim_vals  = [results_sim[m]  * 100 for m in metrics]
    real_vals = [results_real[m] * 100 for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_s = ax.bar(x - width/2, sim_vals,  width,
                    label="Dataset simulé",
                    color="steelblue", alpha=0.85)
    bars_r = ax.bar(x + width/2, real_vals, width,
                    label="Dataset réel (Aissou 2022)",
                    color="tomato", alpha=0.85)

    for bar in list(bars_s) + list(bars_r):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Valeur (%)", fontsize=12)
    ax.set_title("Comparaison des performances\n"
                 "Dataset simulé vs Dataset réel "
                 "(Aissou et al., 2022)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Comparaison sauvegardée : {save_path}")


def print_summary(results_sim, results_real):

    print("\n" + "=" * 55)
    print("  TABLEAU COMPARATIF FINAL")
    print("=" * 55)
    print(f"{'Métrique':<22} {'Simulé':>12} {'Réel':>12}")
    print("-" * 55)

    metrics = [
        ("Accuracy",           "accuracy"),
        ("Taux de détection",  "detection_rate"),
        ("Taux fausse alarme", "false_alarm"),
        ("Précision",          "precision"),
        ("AUC",                "auc"),
    ]
    for label, key in metrics:
        vs = results_sim[key]  * 100
        vr = results_real[key] * 100
        print(f"{label:<22} {vs:>11.2f}% {vr:>11.2f}%")

    print("=" * 55)


if __name__ == "__main__":

    print("=" * 55)
    print("  ÉVALUATION COMPLÈTE — Simulé vs Réel")
    print("=" * 55)

    # 1 — Modèle
    print("\n[1/5] Construction du modèle...")
    model, inference = build_model()

    # 2 — Simulé
    print("\n[2/5] Évaluation dataset simulé...")
    df_sim  = generate_dataset()
    y_proba = predict_proba(inference, df_sim)
    y_pred  = (y_proba > 0.5).astype(int)
    y_true  = df_sim["S"].values
    results_sim = evaluate(y_true, y_pred, y_proba)

    # 3 — Réel
    print("\n[3/5] Évaluation dataset réel...")
    df_real = load_real_dataset()
    y_proba_r = predict_proba(inference, df_real)
    y_pred_r  = (y_proba_r > 0.5).astype(int)
    y_true_r  = df_real["S"].values
    results_real = evaluate(y_true_r, y_pred_r, y_proba_r)

    plot_roc_curve(results_real,
                   save_path="results/roc_curve_real.png")
    plot_confusion_matrix(results_real,
                          save_path="results/confusion_matrix_real.png")
    plot_probability_distribution(results_real,
                          save_path="results/distribution_real.png")

    # 4 — Comparaison datasets
    print("\n[4/5] Comparaison des datasets...")
    compare_datasets(df_sim, df_real)

    # 5 — Résumé
    print("\n[5/5] Résumé final...")
    plot_comparison(results_sim, results_real)
    print_summary(results_sim, results_real)

    print("\n✅ Évaluation terminée !")