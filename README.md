# Détection d'attaques GPS Spoofing pour UAV — Réseaux Bayésiens

**Projet TER — Master 1 RSA (Sécurité)**  
Université / filière : RSA — Sécurité des systèmes embarqués

---

## Contexte

Les drones (UAV) dépendent fortement du GPS pour la navigation autonome. Le **GPS spoofing** consiste à injecter de faux signaux GPS afin de tromper le véhicule sur sa position réelle, ce qui peut provoquer une dérive de trajectoire ou une perte de contrôle.

Ce projet implémente un **réseau bayésien discret** (bibliothèque [pgmpy](https://github.com/pgmpy/pgmpy)) pour estimer la probabilité d'une attaque `P(S=1 | observations)` à partir d'anomalies détectées sur plusieurs capteurs.

Référence du sujet : *Sujet TER M1 RSA — Détection d'attaques GPS Spoofing pour UAV en utilisant les réseaux bayésiens*.

---

## Modèle bayésien

### Structure du graphe (DAG)

```
        S (Spoofing — variable cachée)
       /|\ \
      C D A P   (variables observées, binaires 0/1)
```

| Variable | Signification | Source capteur |
|----------|---------------|----------------|
| **S** | Spoofing (0 = normal, 1 = attaque) | Label / vérité terrain |
| **C** | Incohérence GPS/IMU (via C/N0) | `CN0` — rapport signal/bruit |
| **D** | Anomalie Doppler | `DO` — Doppler Offset |
| **A** | Anomalie d'altitude / qualité position | `PQP` — Position Quality Parameter |
| **P** | Saut de pseudorange | `PD` — Pseudorange Difference |

Le modèle est un **Naïve Bayes inversé** : la cause cachée `S` influence toutes les variables observées, mais les observations sont conditionnellement indépendantes sachant `S`.

### Inférence

- **Moteur** : élimination de variables (`VariableElimination` de pgmpy)
- **Décision** : si `P(S=1 | C, D, A, P) > 0.5` → alerte spoofing
- **CPT** : définies manuellement (simulation) ou apprises par **MLE** sur données réelles

---

## Structure du projet

```
Projet_TER/
│
├── main.py              # Pipeline principal (simulé) — point d'entrée recommandé
├── model.py             # Construction du réseau bayésien discret + CPT manuelles
├── dataset.py           # Génération du dataset simulé (BayesianModelSampling)
├── evaluation.py        # Inférence, métriques (accuracy, AUC, confusion matrix)
├── visualisation.py     # Graphiques ROC, confusion matrix, distributions
│
├── real_dataset.py      # Chargement + discrétisation du dataset réel (Aissou 2022)
├── evaluate_real.py     # Comparaison simulé vs réel (CPT manuelles)
├── evaluate_mle.py      # Comparaison 3 approches (simulé / réel+manuel / réel+MLE)
├── gaussian_bn.py       # Variante Gaussian Naive Bayes (valeurs continues, sans discrétisation)
│
├── requirements.txt     # Dépendances Python
├── data/
│   ├── dataset.csv      # Dataset simulé exporté (2000 lignes)
│   └── GPS_Data_Simplified_2D_Feature_Map.xlsx   # Dataset réel (non versionné, ~69 Mo)
└── results/             # Sorties générées (graphiques PNG, results.txt)
```

### Rôle de chaque fichier

| Fichier | Description |
|---------|-------------|
| `main.py` | Enchaîne les 4 étapes : modèle → dataset simulé → évaluation → visualisations. |
| `model.py` | Définit le DAG `S → {C,D,A,P}` et les tables de probabilité conditionnelle (CPT). Expose `build_model()`. |
| `dataset.py` | Génère 2000 échantillons équilibrés via rejection sampling pgmpy. Expose `generate_dataset()`. |
| `evaluation.py` | Calcule `P(S=1)` pour chaque ligne, puis accuracy, taux de détection, fausses alarmes, AUC. |
| `visualisation.py` | Produit les courbes ROC, matrices de confusion et histogrammes dans `results/`. |
| `real_dataset.py` | Lit le fichier Excel Aissou, calcule des seuils optimaux (Youden), discrétise en `{S,C,D,A,P}`. |
| `evaluate_real.py` | Évalue le même modèle discret sur données simulées **et** réelles ; graphique comparatif. |
| `evaluate_mle.py` | Compare 3 approches : simulé+manuel, réel+manuel, réel+CPT apprises par MLE (`DiscreteMLE`). |
| `gaussian_bn.py` | Alternative avec `GaussianNB` (sklearn) sur features continues — pas de discrétisation. |

---

## Prérequis

- **Python 3.10+** (testé avec Python 3.10 sous Windows)
- **Git** ([git-scm.com](https://git-scm.com/download/win))
- Compte **GitHub**

Vérifier Python :

```powershell
python --version
```

---

## Installation

### 1. Cloner ou ouvrir le projet

```powershell
cd D:\Projet_TER
```

### 2. (Recommandé) Créer un environnement virtuel

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloque l'activation :  
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. Installer les dépendances

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dépendances principales : `pgmpy`, `numpy`, `pandas`, `openpyxl`, `matplotlib`, `scikit-learn`, `seaborn`.

### 4. Dataset réel (obligatoire pour les scripts « réel »)

Le fichier Excel **n'est pas inclus sur GitHub** (69 Mo). Placez-le manuellement :

```
data/GPS_Data_Simplified_2D_Feature_Map.xlsx
```

Source : dataset **Aissou et al. (2022)** — *GPS Data Simplified 2D Feature Map* (attacks GPS spoofing).

Sans ce fichier, seuls `main.py`, `model.py`, `dataset.py` et `evaluation.py` fonctionneront (partie simulée).

---

## Utilisation

### Pipeline complet (données simulées)

```powershell
python main.py
```

Génère : `data/dataset.csv`, `results/results.txt`, graphiques dans `results/`.

### Vérifier le modèle seul

```powershell
python model.py
```

Affiche les scénarios de validation (section 4.5 du rapport) :
- `P(S=1 | C=1) ≈ 0.486`
- `P(S=1 | C=1, D=1) ≈ 0.957`
- `P(S=1 | C=1, D=1, A=1, P=1) ≈ 1.000`

### Autres scripts

```powershell
# Dataset simulé seul
python dataset.py

# Évaluation simulée seule
python evaluation.py

# Préparer le dataset réel
python real_dataset.py

# Simulé vs réel
python evaluate_real.py

# Comparaison 3 approches (simulé / réel+manuel / réel+MLE)
python evaluate_mle.py

# Gaussian BN sur données continues
python gaussian_bn.py
```

### Résultats typiques (indicatifs)

| Approche | Accuracy | Détection | Fausse alarme | AUC |
|----------|----------|-----------|---------------|-----|
| Simulé + CPT manuelle | ~94 % | ~90 % | ~0,6 % | ~0,99 |
| Réel + CPT manuelle | ~60 % | ~66 % | ~45 % | ~0,62 |
| Réel + CPT MLE | ~60 % | ~73 % | ~52 % | ~0,64 |
| Gaussian BN (continu) | variable | — | — | — |

L'écart simulé/réel s'explique par la discrétisation des features continues et la faible séparabilité de certaines variables (D, A) sur le dataset Aissou.

---

## Publier sur GitHub (guide pas à pas)

### Étape 1 — Initialiser Git localement

```powershell
cd D:\Projet_TER

git init
git branch -M main
```

### Étape 2 — Vérifier ce qui sera commité

```powershell
git status
```

Le fichier `.gitignore` exclut :
- l'environnement virtuel `.venv/`
- le cache Python `__pycache__/`
- le fichier Excel volumineux `data/*.xlsx`

Le petit fichier `data/dataset.csv` et les graphiques `results/` **peuvent** être versionnés.

### Étape 3 — Premier commit

```powershell
git add .
git commit -m "Initial commit — détection GPS spoofing par réseau bayésien"
```

### Étape 4 — Créer le dépôt sur GitHub

1. Aller sur [github.com/new](https://github.com/new)
2. Nom du dépôt : par ex. `ter-gps-spoofing-bayesian`
3. Visibilité : **Public** ou **Private**
4. **Ne pas** cocher « Add a README » (déjà présent localement)
5. Cliquer **Create repository**

### Étape 5 — Lier le dépôt distant et pousser

Remplacer `VOTRE_USERNAME` et `NOM_DU_REPO` :

```powershell
git remote add origin https://github.com/VOTRE_USERNAME/NOM_DU_REPO.git
git push -u origin main
```

GitHub demandera vos identifiants. Utilisez un **Personal Access Token** (PAT) à la place du mot de passe :
- GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Cocher au minimum `repo`

Alternative SSH :

```powershell
git remote add origin git@github.com:VOTRE_USERNAME/NOM_DU_REPO.git
git push -u origin main
```

### Étape 6 — Commits suivants

Après chaque modification :

```powershell
git add .
git commit -m "Description concise du changement"
git push
```

### Bonnes pratiques

- Ne jamais committer de secrets (clés API, mots de passe)
- Documenter dans le README comment obtenir le dataset Excel
- Si le rapport LaTeX/PDF est ajouté, le placer dans un dossier `docs/` ou `rapport/`

---

## Livrables TER (rappel du sujet)

| Livrable | Emplacement dans ce repo |
|----------|--------------------------|
| Modèle bayésien implémenté | `model.py`, `gaussian_bn.py` |
| Jeux de données simulés | `dataset.py` → `data/dataset.csv` |
| Expérimentations / métriques | `evaluation.py`, `evaluate_*.py`, `results/` |
| Analyse comparative | `evaluate_real.py`, `evaluate_mle.py` |
| Rapport scientifique (25–35 p.) | À rédiger séparément (hors repo ou dans `docs/`) |

---

## Références

- Sujet TER : Master 1 RSA — Détection GPS Spoofing par réseaux bayésiens
- Dataset réel : Aissou et al. (2022) — GPS spoofing detection features
- Bibliothèque : [pgmpy](https://pgmpy.org/) — Probabilistic Graphical Models in Python

---

## Auteurs

Projet TER — Master RSA Sécurité.
