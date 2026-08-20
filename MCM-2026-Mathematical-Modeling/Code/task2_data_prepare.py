import os
import pandas as pd
import numpy as np


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    step1_path = os.path.join(base_dir, "dwts_step1_processed.csv")
    fan_path = os.path.join(base_dir, "Final_Fan_Votes_Estimated.csv")
    output_path = os.path.join(base_dir, "task2_prepared.csv")

    raw = pd.read_csv(step1_path)
    fan = pd.read_csv(fan_path)

    raw.columns = [c.lower().strip() for c in raw.columns]
    fan.columns = [c.lower().strip() for c in fan.columns]

    fan = fan.rename(columns={
        "season": "season",
        "week": "week",
        "contestant": "contestant",
        "fan_vote_mean": "fan_vote_mean",
        "fan_vote_std": "fan_vote_std",
        "ci_lower_95": "ci_lower_95",
        "ci_upper_95": "ci_upper_95",
        "method": "fan_method"
    })

    merge_cols = ["season", "week", "contestant"]
    merged = pd.merge(raw, fan, on=merge_cols, how="left")

    missing = merged["fan_vote_mean"].isna().sum()
    total = len(merged)
    print(f"合并完成: 总行数={total}, 缺失Fan_Vote_Mean={missing}")

    group_sum = merged.groupby(["season", "week"])["fan_vote_mean"].transform("sum")
    merged["fan_vote_norm"] = np.where(group_sum > 0, merged["fan_vote_mean"] / group_sum, 0)

    merged["judge_rank"] = merged.groupby(["season", "week"])["judge_score_raw"].rank(
        method="min", ascending=False
    )

    merged = merged.sort_values(["season", "week", "contestant"])
    merged.to_csv(output_path, index=False)
    print(f"输出完成: {output_path}")


if __name__ == "__main__":
    main()
