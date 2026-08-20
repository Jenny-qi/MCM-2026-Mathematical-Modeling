import os
import pandas as pd
import numpy as np


def spearman_corr(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    ar = a.rank(method="min")
    br = b.rank(method="min")
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def parse_set(s):
    if pd.isna(s) or s == "":
        return set()
    return set([x for x in s.split(";") if x])


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "task2_week_summary.csv")
    detail_path = os.path.join(base_dir, "task2_week_contestant_scores.csv")
    diff_path = os.path.join(base_dir, "task2_diff_summary.csv")
    corr_path = os.path.join(base_dir, "task2_method_correlation.csv")
    controversy_path = os.path.join(base_dir, "task2_controversy_summary.csv")

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)

    summary["actual_elim_set"] = summary["actual_eliminated"].apply(parse_set)
    summary["rank_pred_set"] = summary["rank_pred_candidates"].apply(parse_set)
    summary["percent_pred_set"] = summary["percent_pred_candidates"].apply(parse_set)

    elim_weeks = summary[summary["actual_eliminated"].fillna("") != ""]

    def match_rate(pred_col, use_save=False):
        if use_save:
            matches = elim_weeks.apply(lambda r: r["actual_eliminated"] == r[pred_col], axis=1)
        else:
            matches = elim_weeks.apply(lambda r: len(r["actual_elim_set"] & r[pred_col]) > 0, axis=1)
        return matches.mean() if len(matches) > 0 else 0.0

    rank_match = match_rate("rank_pred_set", use_save=False)
    percent_match = match_rate("percent_pred_set", use_save=False)
    rank_save_match = match_rate("rank_pred_with_save", use_save=True)
    percent_save_match = match_rate("percent_pred_with_save", use_save=True)

    def diff_rate(set_col_a, set_col_b):
        diffs = summary.apply(lambda r: r[set_col_a] != r[set_col_b], axis=1)
        return diffs.mean()

    rank_percent_diff = diff_rate("rank_pred_set", "percent_pred_set")
    rank_save_percent_save_diff = diff_rate("rank_pred_with_save", "percent_pred_with_save")

    diff_summary = pd.DataFrame([
        {
            "metric": "rank_match_rate",
            "value": rank_match
        },
        {
            "metric": "percent_match_rate",
            "value": percent_match
        },
        {
            "metric": "rank_with_save_match_rate",
            "value": rank_save_match
        },
        {
            "metric": "percent_with_save_match_rate",
            "value": percent_save_match
        },
        {
            "metric": "rank_vs_percent_diff_rate",
            "value": rank_percent_diff
        },
        {
            "metric": "rank_save_vs_percent_save_diff_rate",
            "value": rank_save_percent_save_diff
        }
    ])

    corr_rows = []
    for (season, week), g in detail.groupby(["season", "week"]):
        corr_rows.append({
            "season": season,
            "week": week,
            "corr_fan_rank_vs_total_rank": spearman_corr(g["fan_rank"], g["total_rank"]),
            "corr_fan_rank_vs_total_percent": spearman_corr(g["fan_rank"], -g["total_percent"]),
            "corr_judge_rank_vs_total_rank": spearman_corr(g["judge_rank"], g["total_rank"]),
            "corr_judge_rank_vs_total_percent": spearman_corr(g["judge_rank"], -g["total_percent"])
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_summary = corr_df[[
        "corr_fan_rank_vs_total_rank",
        "corr_fan_rank_vs_total_percent",
        "corr_judge_rank_vs_total_rank",
        "corr_judge_rank_vs_total_percent"
    ]].mean().reset_index()
    corr_summary.columns = ["metric", "value"]

    controversy_names = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]
    controversy_rows = []

    for name in controversy_names:
        g = detail[detail["contestant"] == name]
        if g.empty:
            controversy_rows.append({
                "contestant": name,
                "season": "",
                "actual_elim_week": "",
                "rank_pred_week": "",
                "rank_save_pred_week": "",
                "percent_pred_week": "",
                "percent_save_pred_week": ""
            })
            continue

        for season, sg in g.groupby("season"):
            actual_elim_week = summary[(summary["season"] == season)].apply(
                lambda r: name in r["actual_elim_set"], axis=1
            )
            actual_week = summary[(summary["season"] == season)][actual_elim_week]
            actual_week = int(actual_week["week"].min()) if not actual_week.empty else ""

            rank_week = summary[(summary["season"] == season)].apply(
                lambda r: name in r["rank_pred_set"], axis=1
            )
            rank_week = summary[(summary["season"] == season)][rank_week]
            rank_week = int(rank_week["week"].min()) if not rank_week.empty else ""

            rank_save_week = summary[(summary["season"] == season) & (summary["rank_pred_with_save"] == name)]
            rank_save_week = int(rank_save_week["week"].min()) if not rank_save_week.empty else ""

            percent_week = summary[(summary["season"] == season)].apply(
                lambda r: name in r["percent_pred_set"], axis=1
            )
            percent_week = summary[(summary["season"] == season)][percent_week]
            percent_week = int(percent_week["week"].min()) if not percent_week.empty else ""

            percent_save_week = summary[(summary["season"] == season) & (summary["percent_pred_with_save"] == name)]
            percent_save_week = int(percent_save_week["week"].min()) if not percent_save_week.empty else ""

            controversy_rows.append({
                "contestant": name,
                "season": season,
                "actual_elim_week": actual_week,
                "rank_pred_week": rank_week,
                "rank_save_pred_week": rank_save_week,
                "percent_pred_week": percent_week,
                "percent_save_pred_week": percent_save_week
            })

    pd.DataFrame(controversy_rows).to_csv(controversy_path, index=False)
    diff_summary.to_csv(diff_path, index=False)
    corr_summary.to_csv(corr_path, index=False)
    print(f"输出完成: {diff_path}")
    print(f"输出完成: {corr_path}")
    print(f"输出完成: {controversy_path}")


if __name__ == "__main__":
    main()
