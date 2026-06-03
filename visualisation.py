# visualisation.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from model import build_model
from dataset import generate_dataset
from evaluation import predict_proba, evaluate


def plot_roc_curve(results, save_path="results/roc_curve.png"):
    """
    Biểu đồ 1 — ROC Curve
    Thể hiện trade-off giữa detection rate và false alarm rate
    """

    fpr = results["fpr"]
    tpr = results["tpr"]
    roc_auc = results["auc"]

    plt.figure(figsize=(8, 6))

    # Đường ROC của mô hình
    plt.plot(fpr, tpr,
             color="steelblue",
             linewidth=2,
             label=f"Réseau Bayésien (AUC = {roc_auc:.4f})")

    # Đường random baseline (AUC = 0.5)
    plt.plot([0, 1], [0, 1],
             color="gray",
             linewidth=1,
             linestyle="--",
             label="Classifieur aléatoire (AUC = 0.5)")

    # Điểm tối ưu (threshold = 0.5)
    idx = np.argmin(np.abs(results["y_proba"] - 0.5))
    optimal_fpr = fpr[np.argmin(np.abs(tpr - results["detection_rate"]))]
    optimal_tpr = results["detection_rate"]
    plt.scatter(results["false_alarm"], results["detection_rate"],
                color="red", s=100, zorder=5,
                label=f"Seuil = 0.5 "
                      f"(DR={results['detection_rate']*100:.1f}%, "
                      f"FA={results['false_alarm']*100:.1f}%)")

    plt.xlabel("Taux de fausse alarme (FPR)", fontsize=12)
    plt.ylabel("Taux de détection (TPR)", fontsize=12)
    plt.title("Courbe ROC — Détection GPS Spoofing\n"
              "Réseau Bayésien Naïf (5 variables)",
              fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC Curve sauvegardee : {save_path}")


def plot_confusion_matrix(results, save_path="results/confusion_matrix.png"):
    """
    Biểu đồ 2 — Confusion Matrix
    Thể hiện số lượng TP, TN, FP, FN
    """

    cm = results["confusion_matrix"]

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Prédit Normal", "Prédit Attaque"],
                yticklabels=["Réel Normal", "Réel Attaque"],
                linewidths=0.5,
                linecolor="gray",
                annot_kws={"size": 14})

    plt.title("Matrice de confusion\n"
              f"Accuracy = {results['accuracy']*100:.2f}%",
              fontsize=13)
    plt.ylabel("Valeur réelle", fontsize=11)
    plt.xlabel("Valeur prédite", fontsize=11)


    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix sauvegardee : {save_path}")


def plot_probability_distribution(results, save_path="results/distribution.png"):
    """
    Biểu đồ 3 — Distribution de P(S=1)
    Thể hiện sự phân tách giữa normal et attaque
    """

    y_true  = results["y_true"]
    y_proba = results["y_proba"]

    # Tách xác suất theo nhãn thật
    proba_normal = y_proba[y_true == 0]
    proba_attack = y_proba[y_true == 1]

    plt.figure(figsize=(9, 5))

    # Histogram
    plt.hist(proba_normal, bins=30, alpha=0.6,
             color="steelblue", label="Normal (S=0)",
             edgecolor="white")
    plt.hist(proba_attack, bins=30, alpha=0.6,
             color="tomato", label="Attaque (S=1)",
             edgecolor="white")

    # Đường threshold = 0.5
    plt.axvline(x=0.5, color="black",
                linewidth=2, linestyle="--",
                label="Seuil de décision = 0.5")

    plt.xlabel("P(S=1 | observations)", fontsize=12)
    plt.ylabel("Nombre d'échantillons", fontsize=12)
    plt.title("Distribution des probabilités d'attaque\n"
              "Séparation entre classe normale et classe attaque",
              fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Distribution sauvegardee : {save_path}")


def plot_anomaly_rates(df, save_path="results/anomaly_rates.png"):
    """
    Biểu đồ 4 — Taux d'anomalies par variable
    So sánh tỷ lệ anomalie khi normal vs attaque
    """

    variables = ["C", "D", "A", "P"]
    labels    = [
        "C\n(GPS/IMU)",
        "D\n(Doppler)",
        "A\n(Altitude)",
        "P\n(Pseudorange)"
    ]

    # Tính tỷ lệ anomalie
    rates_normal = [df[df["S"]==0][v].mean()*100 for v in variables]
    rates_attack = [df[df["S"]==1][v].mean()*100 for v in variables]

    # Giá trị CPT kỳ vọng (đường ngang)
    cpt_normal = [5, 3, 2, 4]
    cpt_attack = [90, 70, 40, 65]

    x = np.arange(len(variables))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_normal = ax.bar(x - width/2, rates_normal, width,
                         label="Normal (S=0)",
                         color="steelblue", alpha=0.8,
                         edgecolor="white")
    bars_attack = ax.bar(x + width/2, rates_attack, width,
                         label="Attaque (S=1)",
                         color="tomato", alpha=0.8,
                         edgecolor="white")

    # Valeurs CPT attendues en pointillés
    for i, (cn, ca) in enumerate(zip(cpt_normal, cpt_attack)):
        ax.hlines(cn, i - width, i,
                  colors="navy", linestyles=":", linewidth=1.5)
        ax.hlines(ca, i, i + width,
                  colors="darkred", linestyles=":", linewidth=1.5)

    # Afficher valeurs sur barres
    for bar in bars_normal:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9)
    for bar in bars_attack:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Variable observée", fontsize=12)
    ax.set_ylabel("Taux d'anomalie (%)", fontsize=12)
    ax.set_title("Taux d'anomalies par variable\n"
                 "Comparaison normal vs attaque "
                 "(pointillés = valeurs CPT attendues)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Anomaly Rates sauvegardees : {save_path}")


# ─────────────────────────────────────────
# CHẠY KHI GỌI TRỰC TIẾP
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("=== Génération des visualisations ===\n")

    # Bước 1 — Xây dựng mô hình
    print("Étape 1 : Construction du modèle...")
    model, inference = build_model()

    # Bước 2 — Tạo dataset
    print("\nÉtape 2 : Génération du dataset...")
    df = generate_dataset()

    # Bước 3 — Inference
    print("\nÉtape 3 : Inférence...")
    y_proba = predict_proba(inference, df)
    y_pred  = (y_proba > 0.5).astype(int)
    y_true  = df["S"].values

    # Bước 4 — Evaluation
    print("\nÉtape 4 : Évaluation...")
    results = evaluate(y_true, y_pred, y_proba)

    # Bước 5 — Visualisations
    print("\nÉtape 5 : Génération des graphiques...")
    plot_roc_curve(results)
    plot_confusion_matrix(results)
    plot_probability_distribution(results)
    plot_anomaly_rates(df)

    print("\nToutes les visualisations sont sauvegardees dans results/")