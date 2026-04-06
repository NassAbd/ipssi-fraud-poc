import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # POC : Détection de Fraude Transactionnelle (PIS)
    *Mémoire académique : L'IA au service de l'Open Finance*

    Ce prototype compare deux approches pour la détection de fraudes dans un flux de paiement :
    1. **Modèle A (Legacy)** : Un système expert basé sur des règles déterministes.
    2. **Modèle B (IA)** : Un classifieur Random Forest entraîné sur l'expérience passée.

    Dataset: *https://www.kaggle.com/datasets/ealaxi/banksim1?select=bs140513_032310.csv*
    """)
    return


@app.cell
def _():
    # Installation des dépendances (déjà fait dans l'environnement uv)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score

    sns.set_theme(style="whitegrid", palette="muted")
    return (
        LabelEncoder,
        RandomForestClassifier,
        StandardScaler,
        classification_report,
        confusion_matrix,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Préparation des Données (Data Engineering)
    Nous utilisons le dataset **BankSim**. Les colonnes inutiles sont supprimées, les chaînes sont nettoyées, et les variables catégorielles sont encodées.
    """)
    return


@app.cell
def _(pd):
    # PATH du dataset
    DATA_PATH = "notebook/bs140513_032310.csv"

    def load_data(path):
        # Chargement et nettoyage rapide pour le POC
        try:
           df = pd.read_csv(path)
           # Nettoyage des guillemets
           for col in df.columns:
               if df[col].dtype == 'object':
                   df[col] = df[col].str.replace("'", "")

           # Suppression des colonnes constantes
           df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)
           return df
        except FileNotFoundError:
           # Création d'un mini-dataframe factice pour la démonstration de structure
           print("Dataset non trouvé. Vérifez le chemin")

    df_raw = load_data(DATA_PATH)
    return (df_raw,)


@app.cell
def _(df_raw):
    df_raw.head()
    return


@app.cell
def _(LabelEncoder, StandardScaler, df_raw, train_test_split):
    # --- 1. SÉPARATION INITIALE (Split avant toute transformation) ---
    # On exclut 'customer' et 'merchant' qui sont des IDs uniques (risque d'overfitting)
    # On garde 'step', 'age', 'gender', 'category' et 'amount' comme variables prédictives
    features_list = ['step', 'age', 'gender', 'category', 'amount']

    X_raw = df_raw[features_list]
    y = df_raw['fraud']

    # Split 70/30 avec stratification sur la fraude (indispensable car données déséquilibrées)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42
    )

    # --- 2. ENCODAGE ET PRÉPARATION (Après le split) ---
    # On crée des copies pour ne pas modifier les datasets bruts (utile pour ton Modèle A)
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Encodage des variables catégorielles (Age, Gender, Category)
    categorical_cols = ['age', 'gender', 'category']

    for col in categorical_cols:
        le = LabelEncoder()
        # On apprend les catégories uniquement sur le Train
        X_train[col] = le.fit_transform(X_train_raw[col])
        # On applique au Test (en gérant les éventuelles valeurs inconnues)
        X_test[col] = X_test_raw[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Normalisation du montant
    scaler = StandardScaler()
    X_train['amount'] = scaler.fit_transform(X_train[['amount']])
    X_test['amount'] = scaler.transform(X_test[['amount']])
    return X_test, X_test_raw, X_train, features_list, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Modèles de Détection

    ### Modèle A : Système Expert (Baseline)
    Basé sur une règle métier classique :
    - Blocage au-dessus de 1000€ systématiquement.
    - Blocage au-dessus de 500€ pour certaines catégories à risque ('leisure', 'travel').
    """)
    return


@app.cell
def _(X_test_raw):
    def predict_legacy(row):
        if row['amount'] > 1000:
            return 1
        if row['amount'] > 500 and row['category'] in ['es_leisure', 'es_travel']:
            return 1
        return 0

    y_pred_legacy = X_test_raw.apply(predict_legacy, axis=1)
    return (y_pred_legacy,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Modèle B : IA (Random Forest)
    Utilisation d'une forêt aléatoire avec gestion du déséquilibre des classes (`class_weight='balanced'`).
    """)
    return


@app.cell
def _(RandomForestClassifier, X_train, y_train):

    # --- 3. ENTRAÎNEMENT DU MODÈLE ---
    # Initialisation du Random Forest avec les paramètres optimisés pour le POC
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', # Crucial : donne plus de poids aux fraudes rares
        random_state=42,
        max_depth=10,             # Limite la complexité pour la lisibilité de l'arbre
        n_jobs=-1                 # Utilise tous les processeurs pour accélérer
    )

    print("Entraînement du modèle en cours...")
    rf_model.fit(X_train, y_train)

    # --- 4. VÉRIFICATION DES FEATURES (Optionnel pour debug) ---
    # X_test_raw contient toujours les données d'origine pour ton "Modèle A" (règles métier)
    # X_test contient les données encodées/scalées pour ton "Modèle B" (IA)
    print("Préparation terminée. Prêt pour l'évaluation des modèles.")
    return (rf_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Évaluation et Analyse Comparative
    Nous comparons les performances sur le set de test.
    """)
    return


@app.cell
def _(classification_report, confusion_matrix, y_pred_legacy, y_test):
    # from sklearn.metrics import confusion_matrix, classification_report

    print("--- Modèle A (Legacy) ---")
    report_a = classification_report(y_test, y_pred_legacy)
    print(report_a)
    conf_mat_a = confusion_matrix(y_test, y_pred_legacy)
    print("Confusion Matrix:")
    print(conf_mat_a)
    return


@app.cell
def _(X_test, classification_report, confusion_matrix, rf_model, y_test):
    y_pred_rf = rf_model.predict(X_test)
    report_b = classification_report(y_test, y_pred_rf)
    print("--- Modèle B (Random Forest) ---")
    print(report_b)
    conf_mat_b = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix:")
    print(conf_mat_b)
    return (y_pred_rf,)


@app.cell
def _(confusion_matrix, plt, sns, y_pred_legacy, y_pred_rf, y_test):
    # Exemple à insérer dans une cellule
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_legacy), annot=True, fmt='d', ax=ax[0], cmap='Blues')
    ax[0].set_title('Matrice : Système Expert (Legacy)')
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1], cmap='Greens')
    ax[1].set_title('Matrice : IA (Random Forest)')
    plt.show()
    return


@app.cell
def _(features_list, pd, rf_model):
    importances = pd.Series(rf_model.feature_importances_, index=features_list)
    importances.sort_values().plot(kind='barh', title='Importance des variables pour le Random Forest')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Analyse de l'Impact Business
    Calcul des gains en termes de Faux Positifs évités (meilleure UX) et de fraudes capturées (Sécurité).
    """)
    return


if __name__ == "__main__":
    app.run()
