"""
Visualization utilities.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df: pd.DataFrame, target: str):
    sns.histplot(df[target].dropna(), kde=True, bins=30)
    plt.title(f"Distribution: {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.show()

def correlation_heatmap(df: pd.DataFrame, top_k: int = 20, target: str = None):
    corr = df.select_dtypes("number").corr()
    if target and target in corr.columns:
        cols = corr[target].abs().sort_values(ascending=False).head(top_k).index.tolist()
        corr = df[cols].corr()
    sns.heatmap(corr, annot=False, cmap="RdYlGn", center=0)
    plt.title("Correlation Heatmap")
    plt.show()
