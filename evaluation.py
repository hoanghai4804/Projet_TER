# evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score
)
from model import build_model
from dataset import generate_dataset
from pgmpy.inference import VariableElimination


def predict_proba(inference, df):
    """
    Exécute l'inférence sur l'ensemble du dataset.
    Retourne la probabilité P(S=1) pour chaque ligne.
    """

    probas = []  # Liste des probabilités P(S=1)

    for i, row in df.iterrows():

        # Lire les 4 variables observées de cette ligne
        evidence = {
            "C": int(row["C"]),
            "D": int(row["D"]),
            "A": int(row["A"]),
            "P": int(row["P"])
        }

        # Interroger le modèle : P(S | C=?, D=?, A=?, P=?)
        result = inference.query(
            variables=["S"],
            evidence=evidence,
            show_progress=False
        )

        # Récupérer P(S=1) — probabilité d'attaque
        prob_attack = result.values[1]
        probas.append(prob_attack)

        # Afficher la progression toutes les 200 lignes
        if (i + 1) % 200 == 0:
            print(f"   Traitement : {i+1}/{ len(df)} échantillons...")

    return np.array(probas)


def evaluate(y_true, y_pred, y_proba):
    """
    Calcule toutes les métriques d'évaluation du modèle.

    y_true  → vraies étiquettes (S réel)
    y_pred  → étiquettes prédites (0 ou 1)
    y_proba → probabilité P(S=1) pour chaque ligne
    """

    print("\n=== Résultats de l'évaluation ===\n")

    # ── Accuracy ──────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy           : {acc*100:.2f}%")

    # ── Confusion Matrix ──────────────────────
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]  # Vrai Négatif : normal, prédit normal
    FP = cm[0][1]  # Faux Positif : normal, prédit attaque
    FN = cm[1][0]  # Faux Négatif : attaque, prédit normal
    TP = cm[1][1]  # Vrai Positif : attaque, prédit attaque

    print(f"\nMatrice de confusion :")
    print(f"   Vrai Négatif  (TN) : {TN}")
    print(f"   Faux Positif  (FP) : {FP}")
    print(f"   Faux Négatif  (FN) : {FN}")
    print(f"   Vrai Positif  (TP) : {TP}")

    # ── Metrics clés ──────────────────────────
    detection_rate  = TP / (TP + FN)  # Taux de détection (Recall)
    false_alarm     = FP / (FP + TN)  # Taux de fausse alarme
    precision       = TP / (TP + FP)  # Précision

    print(f"\nTaux de détection  : {detection_rate*100:.2f}%")
    print(f"Taux fausse alarme : {false_alarm*100:.2f}%")
    print(f"Précision          : {precision*100:.2f}%")

    # ── ROC AUC ───────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC                : {roc_auc:.4f}")

    # ── Rapport complet ───────────────────────
    print("\nRapport de classification :")
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal (S=0)", "Attaque (S=1)"]
    ))

    return {
        "accuracy"       : acc,
        "detection_rate" : detection_rate,
        "false_alarm"    : false_alarm,
        "precision"      : precision,
        "auc"            : roc_auc,
        "confusion_matrix": cm,
        "fpr"            : fpr,
        "tpr"            : tpr,
        "y_true"         : y_true,
        "y_pred"         : y_pred,
        "y_proba"        : y_proba
    }


def save_results(results, path="results/results.txt"):
    """Sauvegarde les résultats dans un fichier texte."""
    import os
    os.makedirs("results", exist_ok=True)

    with open(path, "w") as f:
        f.write("=== Résultats TER — Détection GPS Spoofing ===\n\n")
        f.write(f"Accuracy           : {results['accuracy']*100:.2f}%\n")
        f.write(f"Taux de détection  : {results['detection_rate']*100:.2f}%\n")
        f.write(f"Taux fausse alarme : {results['false_alarm']*100:.2f}%\n")
        f.write(f"Précision          : {results['precision']*100:.2f}%\n")
        f.write(f"AUC                : {results['auc']:.4f}\n")

    print(f"\nResultats sauvegardes : {path}")


# ─────────────────────────────────────────
# EXÉCUTION DIRECTE DE CE FICHIER
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("Étape 1 : Construction du modèle...")
    model, inference = build_model()
    print("   Modele valide")

    print("\nÉtape 2 : Génération du dataset...")
    df = generate_dataset()
    print(f"   Dataset : {len(df)} échantillons")

    print("\nÉtape 3 : Inférence sur le dataset...")
    y_proba = predict_proba(inference, df)

    # Seuil de décision = 0.5
    # Si P(S=1) > 0.5 → prédit comme attaque
    y_pred = (y_proba > 0.5).astype(int)
    y_true = df["S"].values

    print("\nÉtape 4 : Évaluation...")
    results = evaluate(y_true, y_pred, y_proba)

    save_results(results)