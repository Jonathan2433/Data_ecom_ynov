import pandas as pd
from datetime import datetime
import numpy as np
import yaml
from datetime import timedelta

class ResellerPrioritizer:
    """
    Class to prioritize resellers based on certain KPIs such as bad reviews and sales value.

    Attributes:
    reviews (DataFrame): DataFrame containing reviews data.
    orders (DataFrame): DataFrame containing orders data.
    items (DataFrame): DataFrame containing items data.
    sellers (DataFrame): DataFrame containing sellers data.
    payments (DataFrame): DataFrame containing payments data.
    result_data (DataFrame): DataFrame containing the final prioritized list of resellers.
    days_back (int): Number of day pass to extract
    current_date (str): date of the day
    """

    def __init__(self, config_path: str, reviews_path: str, orders_path: str, items_path: str,
                 sellers_path: str, payments_path: str, days_back: int, current_date: str) -> None:
        """
        Initialize the ResellerPrioritizer with data paths.

        Parameters:
        reviews_path (str): Path to the reviews CSV file.
        orders_path (str): Path to the orders CSV file.
        items_path (str): Path to the items CSV file.
        sellers_path (str): Path to the sellers CSV file.
        payments_path (str): Path to the payments CSV file.
        """

        # Load the business rules from the YAML file
        with open('./business_rules.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        self.reviews = pd.read_csv(reviews_path)
        self.orders = pd.read_csv(orders_path, parse_dates=['order_delivered_carrier_date'])
        self.items = pd.read_csv(items_path)
        self.sellers = pd.read_csv(sellers_path)
        self.payments = pd.read_csv(payments_path, sep=';')
        self.result_data = None
        self.high_sales_volume_threshold = None  # Ce seuil sera défini plus tard à partir des données
        self.high_total_sales_value_threshold = None  # Ce seuil sera défini plus tard à partir des données
        self.days_back = days_back
        self.current_date = pd.to_datetime(current_date)

    def filter_bad_reviews(self) -> None:
        """Filters reviews to only include those with a score less than 3 and within the specified date range."""
        self.reviews['review_creation_date'] = pd.to_datetime(self.reviews['review_creation_date'])
        cutoff_date = self.current_date - pd.Timedelta(days=self.days_back)
        self.bad_reviews = self.reviews[(self.reviews["review_score"] < 3) &
                                        (self.reviews['review_creation_date'] >= cutoff_date)]

    def merge_data(self) -> None:
        """Merges various datasets based on specific columns for further analysis."""
        merged_data = self.bad_reviews.merge(self.orders, on="order_id", how="inner") \
            .merge(self.items, on="order_id", how="inner") \
            .merge(self.sellers, on="seller_id", how="inner") \
            .merge(self.payments, on="order_id", how="inner")
        self.merged_data = merged_data

    def calculate_priority_score(self):
        """
        Calculate a priority score for each seller based on the client's criteria.
        Sellers with a bad review and comment, high sales volume, and high total sales value are given the highest priority.
        """
        # Calculer les seuils à partir des données
        volume_quantile = self.config['thresholds']['high_sales_volume']
        value_quantile = self.config['thresholds']['high_total_sales_value']
        self.high_sales_volume_threshold = self.result_data['total_orders'].quantile(volume_quantile)
        self.high_total_sales_value_threshold = self.result_data['total_payment_value'].quantile(value_quantile)

        # Ajout des critères basés sur ces seuils
        self.result_data['high_sales_volume'] = self.result_data['total_orders'] > self.high_sales_volume_threshold
        self.result_data['high_total_sales_value'] = self.result_data[
                                                         'total_payment_value'] > self.high_total_sales_value_threshold

        # Adding a column to indicate if there is a comment
        self.result_data['has_comment'] = self.result_data['last_review_comment'].apply(
            lambda x: pd.notnull(x) and x != '')

        # Calculating the priority score
        conditions = [
            (self.result_data['has_comment'] & self.result_data['high_sales_volume'] & self.result_data[
                'high_total_sales_value']),
            (self.result_data['has_comment'] & ~self.result_data['high_sales_volume'] & self.result_data[
                'high_total_sales_value']),
            (self.result_data['has_comment'] & self.result_data['high_sales_volume'] & ~self.result_data[
                'high_total_sales_value']),
            (~self.result_data['has_comment'] & self.result_data['high_sales_volume'] & self.result_data[
                'high_total_sales_value']),
            (~self.result_data['has_comment'] & ~self.result_data['high_sales_volume'] & self.result_data[
                'high_total_sales_value']),
            (~self.result_data['has_comment'] & self.result_data['high_sales_volume'] & ~self.result_data[
                'high_total_sales_value'])
        ]

        # Assigning a different score for each condition
        scores = [6, 5, 4, 3, 2, 1]

        self.result_data['priority_score'] = np.select(conditions, scores, default=0)

    def prioritize_resellers(self) -> None:
        """
        Prioritizes resellers based on various KPIs.
q
        - Aggregates data on total sales, total sales value, and average review score.
        - Fetches the latest review and comment for each seller.
        - Merges aggregated data with the latest review and comment.
        """
        if self.merged_data is None:
            self.merge_data()  # Ensure that data is merged before proceeding.

        # Conversion de la colonne 'order_delivered_carrier_date' au format datetime
        self.merged_data['order_delivered_carrier_date'] = pd.to_datetime(self.merged_data['order_delivered_carrier_date'])

        # Conversion des valeurs de payment_value en float
        self.merged_data['payment_value'] = self.merged_data['payment_value'].str.replace(',', '.').astype(float)

        # Groupement par seller_id et agrégation des données
        aggregated_data = self.merged_data.groupby('seller_id').agg({
            'order_id': 'count',
            'payment_value': 'sum',
            'review_score': 'mean',
            'order_delivered_carrier_date': 'max'
        }).rename(columns={'order_id': 'total_orders',
                           'payment_value': 'total_payment_value',
                           'review_score': 'average_review_score',
                           'order_delivered_carrier_date': 'latest_delivery_date'}).reset_index()

        # Obtention de la dernière note, du dernier commentaire et du montant de la dernière commande du dataset filtré (critiques négatives)
        last_reviews = self.merged_data.sort_values('order_delivered_carrier_date').groupby('seller_id').tail(1)
        last_reviews = last_reviews[['seller_id', 'review_score', 'review_comment_message', 'payment_value']]
        last_reviews.columns = ['seller_id', 'last_review_score', 'last_review_comment',
                                'last_negative_order_value']

        self.result_data = aggregated_data.merge(last_reviews, on='seller_id', how='left')

        # N'oubliez pas d'appeler calculate_priority_score avant de trier les données
        self.calculate_priority_score()

        # Filtrer pour ne garder que les revendeurs avec un score de priorité supérieur à 0
        self.result_data = self.result_data[self.result_data['priority_score'] > 0]

        # Filtrer les revendeurs sans critiques négatives récentes
        self.result_data = self.result_data[self.result_data['last_review_score'].notnull()]

        # Trier les revendeurs d'abord par la date de la dernière critique négative, puis par le total des ventes
        self.result_data = self.result_data.sort_values(
            by=['total_payment_value', 'latest_delivery_date'],
            ascending=[False, False]
        )

        # Filtrer pour éliminer les revendeurs dont la dernière critique négative est antérieure à 2018
        cutoff_date = pd.to_datetime(self.config['date_formats']['cutoff_date'])
        self.result_data = self.result_data[self.result_data['latest_delivery_date'] >= cutoff_date]

        # Calling the method to calculate priority scores
        self.calculate_priority_score()

        # Sorting by the new priority score, and then by the latest delivery date
        self.result_data = self.result_data.sort_values(
            by=['priority_score', 'total_payment_value', 'total_orders', 'latest_delivery_date'],
            ascending=[False, False, False, False]
        )

    def export_to_csv(self) -> None:
        """Exports the prioritized list of resellers to a CSV file."""
        filename = f"./DATA/OUT/resellers_to_contact_{datetime.now().strftime(self.config['export_settings']['datetime_format'])}.csv"
        self.result_data.to_csv(filename, index=False, sep=self.config['export_settings']['csv_separator'])
        print(f"Les données ont été exportées vers {filename}")

    def process(self) -> None:
        """
        Main processing method.

        Combines all steps: filters bad reviews, merges data, prioritizes resellers, and exports to CSV.
        """
        self.filter_bad_reviews()
        self.merge_data()
        self.prioritize_resellers()
        self.export_to_csv()