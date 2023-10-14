import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import plotly.express as px


class Viz:

    def __init__(self, df):
        self.df = df

    def one(self, feature, top=20):
        self.describe(feature)
        self.look(feature, top)

    def describe(self, feature):
        elt = self.df[feature]
        len_df = self.df.shape[0]
        elt_nunique = elt.nunique()
        elt_null = elt.isnull().sum()
        type = elt.dtype
        print(f"Feature : {feature}")
        print(f"Type de donnée : {type}")
        print(f"Nombre de données uniques : {elt_nunique} rows sur {len_df} rows soit {(100*elt_nunique)//len_df}%")
        print(f"Nombre de données nulles : {elt_null} rows sur {len_df} rows soit {(100*elt_null)//len_df}%")
        print("---------------------------------------------------------------")

    def look(self, feature, top=20):
        """--------------------------------------------------------
         Shows a distribution graph of feature
         Parameters : feature in String, top
         Returns : None
        ---------------------------------------------------------"""
        df_serie = self.df[feature].value_counts().head(top)
        high = top // 4
        fig, ax = plt.subplots(figsize=(14, high))
        plot = sns.barplot(x=df_serie.values, y=df_serie.index, ax=ax)
        ax.set_title(f'Distribution de {feature}', size=12, weight='bold')
        ax.bar_label(ax.containers[-1], padding=5)
        ax.tick_params(labelsize=8, color='r')
        plt.tight_layout()
        return None

    def pie(self, feature, top=20):
        """--------------------------------------------------------
         Shows a distribution graph of feature
         Parameters : feature in String, top
         Returns : None
        ---------------------------------------------------------"""
        df_serie = self.df[feature].value_counts().head(top)
        fig = px.pie(values=df_serie.values, labels=df_serie.index, title=f'Distribution de {feature}')
        return None

    def plot(self, feature, top=20):
        print(f'\nEléments de {feature} les plus représentés :')
        self.df[feature].value_counts().head(top).plot(kind='bar')

    def hist(self, feature, hue, top=20):
        """--------------------------------------------------------
        Shows a distribution graph of feature with hue
        Parameters : feature in String, hue in String, top
        Returns : None
        ---------------------------------------------------------"""
        temp_df=self.df[[feature, hue]].copy()
        # make the column categorical, using the order of the `value_counts`
        # temp_df[feature] = pd.Categorical(temp_df[feature], temp_df[feature].value_counts(sort=True).index)
        high = top // 4
        fig, ax = plt.subplots(figsize=(14,high))
        plot = sns.histplot(temp_df, y=feature, hue=hue, multiple="stack", edgecolor=".4", linewidth=.5, ax=ax)
        ax.set_title(f'Distribution de {feature} selon {hue}', size=12, weight='bold')
        ax.tick_params(labelsize=8, color='r')
        plt.tight_layout()

    def data_sample(self, df):
        """--------------------------------------------------------
         Shows samples of features from dataset
         Parameters : self
         Returns : df
         ---------------------------------------------------------"""
        print(f'Notre Dataset est composé de {df.shape[0]} Clients et {df.shape[1]} attributs')
        temp_df=self.df.copy()
        samples = []
        for i in temp_df.columns:
            samples.append(str(list(temp_df[i].head(5))))
        distincts = []
        for i in temp_df.columns:
            distincts.append(len(pd.unique(temp_df[i].astype("string"))))
        obs = pd.DataFrame({
            'name' : self.df.columns,
            'type':self.df.dtypes,
            'sample':samples,
            'nb_uniques': distincts,
            '%_uniques':round((temp_df.nunique()/len(temp_df))*100, 2),
            '%_nulls':round((temp_df.isnull().sum()/len(temp_df))*100, 2)
        })
        return obs.reset_index(drop=True)