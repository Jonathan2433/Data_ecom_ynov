## Projet Data Science pour E-commerçant

### Description
Le projet ResellerPrioritizer aide l'entreprises à identifier et prioriser les revendeurs basés sur leurs performances et la qualité des services. 

Il utilise les données des avis clients, des commandes, des articles, des vendeurs et des paiements pour créer une liste prioritaire des revendeurs.

### Fonctionnalités

#### Filtration des Avis Négatifs: 
Sélectionne les revendeurs ayant reçu des avis négatifs dans un intervalle de temps spécifié.
#### Calcul du Score de Priorité: 
Évalue chaque revendeur en fonction de critères tels que le volume de vente, la valeur totale des ventes et la présence de commentaires.
#### Exportation des Données: 
Crée un fichier CSV avec la liste prioritaire des revendeurs.

### Installation
#### Prérequis:

#### Python 3.x: 
Langage de programmation.
#### Pandas: 
Bibliothèque Python pour la manipulation de données.
#### Numpy: 
Bibliothèque Python pour le calcul scientifique.
#### PyYAML: 
Bibliothèque Python pour analyser et générer des fichiers YAML.

### Étapes d'installation:
#### Installation de Python: T
éléchargez et installez Python depuis le site officiel.
#### Installer pip: 
pip est le gestionnaire de paquets pour Python. Il est généralement inclus avec Python.

### Configuration de l'environnement virtuel:

#### Sur macOS/Linux: 
Utilisez python3 -m venv nom_de_lenv et source nom_de_lenv/bin/activate.
#### Sur Windows: 
Utilisez python -m venv nom_de_lenv et .\nom_de_lenv\Scripts\activate.

### Installation des dépendances: 
Dans l'environnement virtuel, installez les dépendances via pip install -r requirements.txt.

#### Explication des Dépendances
#### ipykernel: 
Permet à Jupyter de gérer des cellules Python.
#### notebook: 
Application Jupyter Notebook pour créer et partager des documents contenant du code et des données.
#### pandas: 
Bibliothèque pour la manipulation et l'analyse de données.
#### numpy: 
Bibliothèque pour le calcul scientifique.
#### python-dotenv: 
Gère les variables d'environnement.

### Téléchargement du Code
Téléchargez les fichiers du projet dans un répertoire de votre choix.

### Configuration
Configurez les chemins des fichiers de données et les règles métier dans business_rules.yaml.

### Utilisation
#### Paramétrage des Chemins de Données
Définissez les chemins vers les fichiers CSV des avis, commandes, articles, vendeurs et paiements.

### Exécution du Script
Ouvrez un terminal.
Naviguez vers le répertoire du script.
Exécutez avec python [nom_du_script].py.

### Résultats
Le script génère un fichier CSV dans ./DATA/OUT/ avec la liste des revendeurs.

### Exemple d'Utilisation
from ASSETS.CLASS.ResellerPrioritizer import ResellerPrioritizer

data_paths = {
    "reviews_path": "./DATA/IN/olist_order_reviews_dataset.csv",
    "orders_path": "./DATA/IN/olist_orders_dataset.csv",
    "items_path": "./DATA/IN/olist_order_items_dataset.csv",
    "sellers_path": "./DATA/IN/olist_sellers_dataset.csv",
    "payments_path": "./DATA/IN/olist_order_payments_dataset.csv"
}

config_path = './business_rules.yaml'
days_back = 1
current_date = '2018-09-01'

reseller_manager = ResellerPrioritizer(config_path, **data_paths, days_back=days_back, current_date=current_date)
reseller_manager.process()


### Support
Pour toute question ou assistance, contactez Jonathan Dupau.


## Classe ResellerPrioritizer
La classe ResellerPrioritizer de ce projet a pour but de classer les revendeurs en analysant des indicateurs clés de performance (KPI) tels que le volume des ventes, la valeur des ventes et les scores des avis clients. 

Elle exploite des données issues de multiples sources, y compris les avis, les commandes, les articles, les vendeurs et les paiements, pour construire un profil complet de chaque revendeur.

### Fonctionnalités
### Intégration des données : 
Combine les données de plusieurs fichiers CSV en un seul DataFrame pour l'analyse.
### Analyse des avis : 
Filtre les avis pour identifier ceux avec des scores faibles dans un délai récent, indiquant une performance médiocre.
### Agrégation des données : 
Fusionne les ensembles de données pertinents et agrège les données pour calculer le total des ventes, la valeur totale des ventes et les scores moyens d'avis pour chaque vendeur.
### Calcul du score de priorité : 
Attribue un score de priorité à chaque revendeur basé sur une combinaison d'avis négatifs, de volume de ventes élevé et de grande valeur totale des ventes.
### Compilation des résultats : 
Produit un DataFrame final avec les revendeurs priorisés, prêt pour des actions ou des examens supplémentaires.

## Flux de travail
### Initialisation : 
Accepte les chemins vers les fichiers CSV pertinents et les charge dans des DataFrames, préparant le terrain pour le processus de priorisation.
### Filtrage des avis 
négatifs : Isoler les avis avec des scores en dessous d'un certain seuil qui ont été postés dans un nombre défini de jours à partir de la date actuelle.
### Fusion des données : 
Combine les avis filtrés avec les données des commandes, des articles, des vendeurs et des paiements pour une vue d'ensemble.
### Calcul du score de priorité : 
Utilise des seuils prédéfinis pour identifier les revendeurs à haute priorité et calcule un score de priorité pour chacun.
### Priorisation des revendeurs : 
Agrège et fusionne les données pour mettre en évidence les revendeurs nécessitant une attention, basée sur le score de priorité.
### Exportation des résultats : 
Sort la liste finalisée des revendeurs priorisés dans un fichier CSV, formaté selon les paramètres spécifiés.
### Utilisation
Pour utiliser cette classe, instanciez-la avec les chemins de fichiers CSV requis et les paramètres, puis appelez la méthode process() pour exécuter le flux de travail complet de priorisation. Le résultat sera un fichier CSV contenant la liste triée des revendeurs basée sur le score de priorité calculé, prêt pour un usage commercial tel que des communications ciblées ou des interventions.