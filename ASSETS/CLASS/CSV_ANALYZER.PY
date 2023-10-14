import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("ticks")

class CSVAnalyzer:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def describe(self):
        print(f"CSV : '{file_path}' - {self.df.shape[0]} lignes, {self.df.shape[1]} features")
        print("="*40)
        info = []
        for column in self.df.columns:
            non_nulls = self.df[column].notna().sum()
            sample_value = self.df[column].dropna().iloc[0] if non_nulls > 0 else None
            info.append([column, self.df[column].dtype, self.df.shape[0] - non_nulls, sample_value])

        info_df = pd.DataFrame(info, columns=["Feature", "Type", "NaN values", "Sample value"])
        print(info_df)
        print("="*40)

    def look(self, feature, top=20):
        """Shows a distribution graph of feature"""
        df_serie = self.df[feature].value_counts().head(top)
        high = top // 4
        fig, ax = plt.subplots(figsize=(14, high))
        sns.barplot(x=df_serie.values, y=df_serie.index, ax=ax)
        ax.set_title(f'Distribution de {feature}', size=12, weight='bold')
        ax.bar_label(ax.containers[-1], padding=5)
        ax.tick_params(labelsize=8, color='r')
        plt.tight_layout()
        plt.show()

    # Vous pouvez ajouter d'autres fonctions comme celles de la classe Viz pour enrichir votre analyse.

