# Specs d'implémentation : Prototype de Détection de Fraude (PIS)

## 1. Environnement et Stack Technique
* **Langage :** Python 3.x
* **Librairies clés :** * `pandas`, `numpy` (Manipulation de données)
    * `scikit-learn` (Prétraitement, Random Forest, Métriques)
    * `matplotlib`, `seaborn` (Visualisation des résultats)
* **Source de données :** Dataset BankSim (CSV)

## 2. Pipeline de Traitement des Données (Data Engineering)
L'agent IA doit suivre ces étapes pour préparer le terrain :
1.  **Chargement :** Importer `bs140513_032310.csv`.
2.  **Nettoyage :** * Supprimer les colonnes inutiles (ex: `zipcodeOri`, `zipMerchant` car elles sont constantes dans ce dataset).
    * Supprimer les guillemets simples dans les valeurs de chaînes de caractères.
3.  **Feature Engineering :**
    * **Encodage :** Appliquer un `LabelEncoder` ou `One-Hot Encoding` sur les variables catégorielles (`category`, `gender`).
    * **Normalisation :** Appliquer un `StandardScaler` sur la colonne `amount`.
4.  **Split :** Diviser les données en 70% train / 30% test avec un `stratify=y` pour conserver la proportion de fraudes dans les deux sets.


---

## 3. Implémentation des Modèles

### Modèle A : Système Expert (Baseline "Legacy")
Simuler un moteur de règles bancaires traditionnel sans IA.
* **Logique :** Créer une fonction `predict_rules(row)` :
    * SI `amount > 500` ET `category` est parmi ['leisure', 'travel'] : FRAUDE.
    * SI `amount > 1000` : FRAUDE.
    * SINON : LÉGITIME.

### Modèle B : IA (Random Forest)
* **Configuration :** `RandomForestClassifier` avec 100 estimateurs.
* **Gestion du déséquilibre :** Utiliser le paramètre `class_weight='balanced'` pour que l'IA accorde plus d'importance aux cas de fraude (rares).
* **Entraînement :** Ajuster sur le set d'entraînement.

---

## 4. Évaluation et Comparaison (Livrables visuels)
L'agent doit générer les éléments suivants pour ton mémoire :

1.  **Matrices de Confusion :** Comparaison côte à côte du Modèle A et du Modèle B.
2.  **Rapport de Classification :** Calculer pour les deux modèles :
    * **Précision :** $P = \frac{TP}{TP + FP}$ (Focus sur la réduction des faux positifs).
    * **Recall (Rappel) :** $R = \frac{TP}{TP + FN}$ (Capacité à capturer les fraudes).
    * **F1-Score :** Moyenne harmonique des deux.
3.  **Courbe de Précision-Rappel :** Plus adaptée que la courbe ROC pour les données déséquilibrées (Imbalanced Data).


---

## 5. Script d'Analyse Business (Sortie attendue)
Le code doit produire un résumé textuel pour alimenter ta partie **3.4 Impact Métier** :
* **Gain en UX :** Nombre de clients légitimes qui ne sont plus bloqués par rapport au Modèle A.
* **Efficacité Opérationnelle :** Pourcentage de réduction du volume d'alertes à traiter manuellement.
* **Sécurité :** Nombre de fraudes réelles manquées par l'IA vs le Système Expert.