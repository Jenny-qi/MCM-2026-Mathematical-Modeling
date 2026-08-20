import os
import pandas as pd
import numpy as np


def pick_bottom_candidates(series, mode="max"):
    if mode == "max":
        target = series.max()
        return series[series == target].index.tolist()
    target = series.min()
    return series[series == target].index.tolist()


def rank_method(group):
    g = group.set_index("contestant")
    judge_rank = g["judge_score_raw"].rank(method="min", ascending=False)
    fan_rank = g["fan_vote_norm"].rank(method="min", ascending=False)
    total_rank = judge_rank + fan_rank

    order = g.assign(
        total_rank=total_rank,
        judge_rank=judge_rank,
        fan_rank=fan_rank
    ).sort_values(
        ["total_rank", "judge_score_raw", "fan_vote_norm"],
        ascending=[False, True, True]
    )

    bottom_two = order.head(2).index.tolist()
    if len(bottom_two) == 0:
        bottom_two = []

    if len(bottom_two) == 1:
        elim_with_save = bottom_two[0]
        save_with_save = ""
    else:
        sub = order.loc[bottom_two].sort_values(
            ["judge_score_raw", "fan_vote_norm"],
            ascending=[True, True]
        )
        elim_with_save = sub.index[0]
        save_with_save = sub.index[1]

    candidates = pick_bottom_candidates(total_rank, mode="max")
    return total_rank, judge_rank, fan_rank, candidates, bottom_two, elim_with_save, save_with_save


def percent_method(group):
    g = group.set_index("contestant")
    judge_total = g["judge_score_raw"].sum()
    if judge_total == 0:
        judge_percent = g["judge_score_raw"] * 0
    else:
        judge_percent = g["judge_score_raw"] / judge_total * 100

    fan_total = g["fan_vote_norm"].sum()
    if fan_total == 0:
        fan_percent = g["fan_vote_norm"] * 0
    else:
        fan_percent = g["fan_vote_norm"] / fan_total * 100

    total_percent = judge_percent + fan_percent

    order = g.assign(
        total_percent=total_percent,
        judge_percent=judge_percent,
        fan_percent=fan_percent
    ).sort_values(
        ["total_percent", "judge_score_raw", "fan_vote_norm"],
        ascending=[True, True, True]
    )

    bottom_two = order.head(2).index.tolist()
    if len(bottom_two) == 0:
        bottom_two = []

    if len(bottom_two) == 1:
        elim_with_save = bottom_two[0]
        save_with_save = ""
    else:
        sub = order.loc[bottom_two].sort_values(
            ["judge_score_raw", "fan_vote_norm"],
            ascending=[True, True]
        )
        elim_with_save = sub.index[0]
        save_with_save = sub.index[1]

    candidates = pick_bottom_candidates(total_percent, mode="min")
    return total_percent, judge_percent, fan_percent, candidates, bottom_two, elim_with_save, save_with_save


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "task2_prepared.csv")
    summary_path = os.path.join(base_dir, "task2_week_summary.csv")
    detail_path = os.path.join(base_dir, "task2_week_contestant_scores.csv")

    df = pd.read_csv(input_path)
    df = df.sort_values(["season", "week", "contestant"])

    summaries = []
    details = []

    for (season, week), group in df.groupby(["season", "week"]):
        actual_elim = group.loc[group["is_eliminated"] == True, "contestant"].tolist()
        actual_elim_str = ";".join(sorted(actual_elim))

        total_rank, judge_rank, fan_rank, rank_candidates, rank_bottom_two, rank_elim_save, rank_saved = rank_method(group)
        total_percent, judge_percent, fan_percent, percent_candidates, percent_bottom_two, percent_elim_save, percent_saved = percent_method(group)

        for _, row in group.iterrows():
            name = row["contestant"]
            details.append({
                "season": season,
                "week": week,
                "contestant": name,
                "judge_score_raw": row["judge_score_raw"],
                "fan_vote_norm": row["fan_vote_norm"],
                "judge_rank": judge_rank.loc[name],
                "fan_rank": fan_rank.loc[name],
                "total_rank": total_rank.loc[name],
                "judge_percent": judge_percent.loc[name],
                "fan_percent": fan_percent.loc[name],
                "total_percent": total_percent.loc[name]
            })

        summaries.append({
            "season": season,
            "week": week,
            "actual_system": group.iloc[0]["system"],
            "actual_has_save": bool(group.iloc[0]["has_save"]),
            "actual_eliminated": actual_elim_str,
            "rank_pred_candidates": ";".join(sorted(rank_candidates)),
            "rank_bottom_two": ";".join(rank_bottom_two),
            "rank_pred_with_save": rank_elim_save,
            "rank_judge_saved": rank_saved,
            "percent_pred_candidates": ";".join(sorted(percent_candidates)),
            "percent_bottom_two": ";".join(percent_bottom_two),
            "percent_pred_with_save": percent_elim_save,
            "percent_judge_saved": percent_saved,
            "contestant_count": len(group)
        })

    pd.DataFrame(details).to_csv(detail_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"输出完成: {summary_path}")
    print(f"输出完成: {detail_path}")


if __name__ == "__main__":
    main()
