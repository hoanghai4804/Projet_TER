# evaluate_mle.py
"""
Comparaison 3 approches:
1. Simulated data + CPT manuelle
2. Real data + CPT manuelle (Youden)
3. Real data + CPT apprise par MLE
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination
from model import build_model
from real_dataset import load_real_dataset, compare_datasets
from dataset import generate_dataset
from evaluation import predict_proba, evaluate
from visualisation import plot_roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os


def build_model_mle(df_train):
    """
    Construit un réseau bayésien dont les CPT
    sont apprises depuis les données réelles (MLE).
    Structure DAG identique : S → {C, D, A, P}
    """

    print("\n=== Construction modèle MLE ===")

    # Même structure DAG
    model = DiscreteBayesianNetwork([
        ("S", "C"),
        ("S", "D"),
        ("S", "A"),
        ("S", "P")
    ])

    # Apprendre les CPT depuis les données
    model.fit(df_train, estimator=DiscreteMLE())

    assert model.check_model()
    print("Modele MLE valide")

    # Afficher les CPT apprises
    print("\n=== CPT apprises depuis données réelles ===\n")
    for cpd in model.cpds:
        print(cpd)
        print()

    inference = VariableElimination(model)
    return model, inference


def plot_comparison_3(r_sim, r_manual, r_mle,
                      save_path="results/comparison_3.png"):
    """
    Biểu đồ so sánh 3 approches
    """

    metrics = ["accuracy", "detection_rate",
               "false_alarm", "auc"]
    labels  = ["Accuracy", "Taux détection",
               "Fausse alarme", "AUC"]

    v_sim    = [r_sim[m]    * 100 for m in metrics]
    v_manual = [r_manual[m] * 100 for m in metrics]
    v_mle    = [r_mle[m]    * 100 for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - width, v_sim,    width,
                label="Simulé + CPT manuelle",
                color="steelblue", alpha=0.85)
    b2 = ax.bar(x,          v_manual, width,
                label="Réel + CPT manuelle (Youden)",
                color="tomato", alpha=0.85)
    b3 = ax.bar(x + width,  v_mle,    width,
                label="Réel + CPT MLE",
                color="seagreen", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Valeur (%)", fontsize=12)
    ax.set_title(
        "Comparaison des 3 approches\n"
        "Simulé vs Réel+Manuel vs Réel+MLE",
        fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparaison 3 sauvegardee : {save_path}")


def print_summary_3(r_sim, r_manual, r_mle):

    print("\n" + "=" * 65)
    print("  TABLEAU COMPARATIF — 3 APPROCHES")
    print("=" * 65)
    print(f"{'Métrique':<22} {'Simulé+Manuel':>14} "
          f"{'Réel+Manuel':>13} {'Réel+MLE':>11}")
    print("-" * 65)

    metrics = [
        ("Accuracy",           "accuracy"),
        ("Taux de détection",  "detection_rate"),
        ("Taux fausse alarme", "false_alarm"),
        ("Précision",          "precision"),
        ("AUC",                "auc"),
    ]

    for label, key in metrics:
        vs = r_sim[key]    * 100
        vm = r_manual[key] * 100
        vl = r_mle[key]    * 100
        print(f"{label:<22} {vs:>13.2f}% "
              f"{vm:>12.2f}% {vl:>10.2f}%")

    print("=" * 65)


if __name__ == "__main__":

    print("=" * 65)
    print("  COMPARAISON 3 APPROCHES")
    print("=" * 65)

    # ── 1. Dataset simulé + CPT manuelle ──────────
    print("\n[1/4] Approche 1 — Simulé + CPT manuelle...")
    _, inference_manual = build_model()
    df_sim   = generate_dataset()
    y_proba  = predict_proba(inference_manual, df_sim)
    y_pred   = (y_proba > 0.5).astype(int)
    r_sim    = evaluate(df_sim["S"].values, y_pred, y_proba)

    # ── 2. Dataset réel + CPT manuelle (Youden) ───
    print("\n[2/4] Approche 2 — Réel + CPT manuelle...")
    df_real    = load_real_dataset()
    y_proba_m  = predict_proba(inference_manual, df_real)
    y_pred_m   = (y_proba_m > 0.5).astype(int)
    r_manual   = evaluate(df_real["S"].values,
                          y_pred_m, y_proba_m)

    # ── 3. Dataset réel + CPT MLE ─────────────────
    print("\n[3/4] Approche 3 — Réel + CPT MLE...")

    # Chia train 70% / test 30%
    df_train = df_real.sample(frac=0.7, random_state=42)
    df_test  = df_real.drop(df_train.index)

    print(f"   Train : {len(df_train)} échantillons")
    print(f"   Test  : {len(df_test)}  échantillons")

    # Học CPT từ train data
    _, inference_mle = build_model_mle(df_train)

    # Test trên test data
    y_proba_mle = predict_proba(inference_mle, df_test)
    y_pred_mle  = (y_proba_mle > 0.5).astype(int)
    r_mle       = evaluate(df_test["S"].values,
                           y_pred_mle, y_proba_mle)

    plot_roc_curve(r_mle,
        save_path="results/roc_curve_mle.png")
    plot_confusion_matrix(r_mle,
        save_path="results/confusion_matrix_mle.png")

    # ── 4. Comparaison finale ─────────────────────
    print("\n[4/4] Comparaison finale...")
    plot_comparison_3(r_sim, r_manual, r_mle)
    print_summary_3(r_sim, r_manual, r_mle)

    print("\nTermine ! Resultats dans : results/")