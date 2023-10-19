import pandas as pd
import glob
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import plotly.express as px
from Viz import Viz
import os 
class DataAnalyzer:

    def __init__(self, directory_path):
        self.dataframes = {}
        self._load_dataframes(directory_path)
        
    def _load_dataframes(self, directory_path):
        search_path = os.path.join(directory_path, "*.csv")  # Construisez le chemin avec os.path.join
        print(f"Searching for files in: {search_path}")  # Print le chemin recherché
        for file_path in glob.glob(search_path):
            file_name = os.path.basename(file_path).replace('.csv', '')  # Utilisez os.path.basename
            print(f"Loading {file_name} from {file_path}")  
            self.dataframes[file_name] = pd.read_csv(file_path)

    def describe_all(self):
        for name, df in self.dataframes.items():
            print(f"Analyzing {name}:")
            viz = Viz(df)
            for column in df.columns:
                viz.describe(column)

    def look_all(self, top=20):
        for name, df in self.dataframes.items():
            viz = Viz(df)
            for column in df.columns:
                viz.look(column, top=top)

    def pie_all(self, top=20):
        for name, df in self.dataframes.items():
            viz = Viz(df)
            for column in df.columns:
                viz.pie(column, top=top)
                
    # Vous pouvez ajouter d'autres méthodes en utilisant le même modèle.
    
    def get_dataframe(self, name):
        return self.dataframes.get(name, None)


