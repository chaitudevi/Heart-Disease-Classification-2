import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_class_balance(df, target_col, output_dir):
    logger.info("Creating class balance plot")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title("Target Class Distribution")
    plt.savefig(os.path.join(output_dir, "class_balance.png"))
    plt.close()


def plot_histograms(df, output_dir):
    logger.info("Creating feature histograms")
    df.hist(figsize=(16, 12), bins=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_histograms.png"))
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    logger.info("Creating correlation heatmap")
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
