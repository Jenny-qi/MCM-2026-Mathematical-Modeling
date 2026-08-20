import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    diff_path = os.path.join(base_dir, "task2_diff_summary.csv")
    corr_path = os.path.join(base_dir, "task2_method_correlation.csv")
    out_dir = os.path.join(base_dir, "task2_outputs")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    diff_df = pd.read_csv(diff_path)
    corr_df = pd.read_csv(corr_path)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=diff_df, x="metric", y="value")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rate")
    plt.title("Task2 Match and Difference Rates")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_match_diff_rates.png"))
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=corr_df, x="metric", y="value")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Spearman Correlation")
    plt.title("Fan/Judge Influence Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_correlation_summary.png"))
    plt.close()

    print(f"输出完成: {out_dir}")


if __name__ == "__main__":
    main()
