from reseller_prioritizer import ResellerPrioritizer

# Utilisation :
data_paths = {
    "reviews_path": "./DATA/IN/olist_order_reviews_dataset.csv",
    "orders_path": "./DATA/IN/olist_orders_dataset.csv",
    "items_path": "./DATA/IN/olist_order_items_dataset.csv",
    "sellers_path": "./DATA/IN/olist_sellers_dataset.csv",
    "payments_path": "./DATA/IN/olist_order_payments_dataset.csv"
}

config_path = './business_rules.yaml'
days_back = 1  # Pour extraire les avis de la veille
current_date = '2018-09-01'

# Création de l'instance avec les nouveaux paramètres
reseller_manager = ResellerPrioritizer(config_path, **data_paths, days_back=days_back, current_date=current_date)
reseller_manager.process()
