# Proof of Concept (POC) : Détection de Fraude Transactionnelle dans l'Open Banking (PIS)

**Auteur:** Abdallah NASSUR
---

Ce dépôt contient le code source du cas concret de recherche du mémoire dédié à la **sécurisation de l'initiation de paiement (PIS)**. 

L'objectif de ce Proof of Concept (POC) est de comparer empiriquement deux approches :
1.  **Un Système Expert (Moteur de règles déterministe)** : Utilisant des seuils statiques et des catégories de risques rigides (approche patrimoniale).
2.  **Une Intelligence Artificielle (Machine Learning supervisé)** : Basée sur un classifieur ensembliste *Random Forest* entraîné à gérer le déséquilibre extrême des classes et combiné à un outil d'explicabilité locale (**SHAP**).

---

## 📊 Principaux Résultats du Match

L'évaluation comparative a été menée sur le jeu de données de test **BankSim** ($178\,393$ transactions, dont $2\,160$ fraudes réelles) :

| Métrique | Système Expert (Legacy) | Random Forest (IA) | Impact Opérationnel |
| :--- | :---: | :---: | :---: |
| **Vrais Positifs (TP)** | $283$ | $1\,359$ | **Fraudes manquées divisées par 5** |
| **Faux Positifs (FP)** | $12$ | $224$ | Friction maîtrisée à rappel élevé |
| **Taux de Rappel (Sensitivity)** | $13,10\%$ | $62,92\%$ | Détection robuste des signaux faibles |
| **Précision** | $95,93\%$ | $85,85\%$ | Alertes qualifiées et fiables |
| **Score F1** | $23,06\%$ | $72,62\%$ | Performance globale multipliée par 3 |

---

## 📁 Structure du Projet

```text
├── app/                      # Interface utilisateur Streamlit
│   ├── main.py               # Point d'entrée de l'application
│   ├── pages/
│   │   ├── 1_dashboard.py    # Visualisation des résultats et KPIs
│   │   ├── 2_simulator.py    # Simulateur de transaction temps réel (API mockée + SHAP)
│   │   └── 3_roi.py          # Modélisateur du ROI et de l'impact financier
│   └── styles.py             # Configuration esthétique (Dark Mode Premium)
├── models/                   # Artefacts binaires sérialisés (exclus de Git)
├── notebook/                 # Espace d'exploration et de stockage du dataset BankSim
├── scripts/                  # Scripts utilitaires d'automatisation
│   └── train_and_export.py   # Script d'entraînement optimisé par GridSearchCV
├── src/                      # Code source du package logique principal
│   ├── models/
│   │   ├── expert_system.py  # Logique du Système Expert
│   │   └── ml_model.py       # Wrapper de prédiction de l'IA
│   └── schemas/
│       └── fraud_detection.py# Contrats d'interfaces Pydantic (Schema-First API)
└── tests/                    # Suite de tests unitaires et de robustesse
```

---

## 🚀 Installation & Utilisation

Le projet utilise le gestionnaire de paquets et d'environnements moderne **`uv`** pour garantir l'isolation et la reproductibilité de l'environnement.

### 1. Prérequis
Assurez-vous d'avoir installé Python (3.11 ou supérieur) et `uv`.

### 2. Cloner le projet et installer les dépendances
```bash
# Installer l'environnement virtuel et les packages du projet
uv sync
```

### 3. Télécharger le Dataset
Téléchargez le fichier `bs140513_032310.csv` (BankSim dataset issu de Kaggle) et placez-le dans le répertoire `notebook/`.

### 4. Entraîner le Modèle (Optimisation GridSearchCV)
Exécutez le script d'entraînement pour générer les fichiers joblib de normalisation, d'encodage et le modèle optimisé :
```bash
uv run python scripts/train_and_export.py --data notebook/bs140513_032310.csv
```

### 5. Lancer l'Application Streamlit
Démarrez le tableau de bord interactif pour observer les visualisations, simuler des transactions avec l'interprétabilité SHAP et calculer le ROI financier :
```bash
uv run streamlit run app/main.py
```

### 6. Lancer la Suite de Tests
Validez l'intégrité du pipeline de données, les formes de tenseurs (TDD Shape checks) et la logique des modèles en exécutant :
```bash
uv run pytest tests/
```

---

## 🛠️ Concepts Techniques Implémentés

*   **GridSearchCV & Rééquilibrage** : Optimisation des hyperparamètres du modèle en maximisant le score F1 et utilisation de pénalités de classe (`class_weight='balanced'`) pour compenser le déséquilibre structurel de classes ($0,46\%$ de fraudes).
*   **Schema-First (Pydantic)** : Typage fort et validation rigoureuse des données d'entrée au niveau du simulateur temps réel via des schémas contractuels (`TransactionInput` et `RiskScoreOutput`).
*   **Explicabilité SHAP (TreeExplainer)** : Interprétation mathématique locale de la décision du modèle supervisé pour chaque transaction simulée via l'extraction des contributions des caractéristiques (Shapley values).
*   **Modélisation financière du ROI** : Calculateur dynamique mesurant la réduction nette de l'OPEX (charge de revue manuelle des fausses alertes) et du coût de la friction UX à rappel constant.
