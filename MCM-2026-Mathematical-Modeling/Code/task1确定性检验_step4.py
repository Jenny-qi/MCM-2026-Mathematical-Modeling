import pandas as pd
import numpy as np
from scipy.stats import rankdata

def validate_consistency(estimated_file, raw_processed_file, report_file):
    print("正在执行一致性检验 (Validation)...")
    
    # 读取估算结果和原始规则数据
    est_df = pd.read_csv(estimated_file)
    raw_df = pd.read_csv(raw_processed_file)
    
    # 合并数据
    # 我们需要 raw_df 中的 judge_metric, system, has_save, is_eliminated
    # 以及 est_df 中的 Fan_Vote_Mean
    
    merged = pd.merge(raw_df, est_df[['Season', 'Week', 'Contestant', 'Fan_Vote_Mean']], 
                      left_on=['season', 'week', 'contestant'], 
                      right_on=['Season', 'Week', 'Contestant'], 
                      how='inner')
    
    validation_results = []
    total_weeks = 0
    consistent_weeks = 0
    
    # 按周遍历
    groups = merged.groupby(['season', 'week'])
    
    for (season, week), group in groups:
        # 只验证有淘汰发生的周
        elim_rows = group[group['is_eliminated'] == True]
        if elim_rows.empty:
            continue
            
        total_weeks += 1
        
        # 获取当周规则
        system = group.iloc[0]['system']
        has_save = group.iloc[0]['has_save']
        
        # 准备数据计算
        contestants = group['contestant'].values
        judge_metrics = group['judge_metric'].values # Rank值 或 Percent值
        fan_votes_mean = group['Fan_Vote_Mean'].values # 估算的百分比 (0-1)
        
        # --- 重演比赛逻辑 ---
        
        final_metric = None
        
        if system == 'rank':
            # 1. 将估算的观众百分比转化为排名
            # 票数越高(大) -> Rank数值越小(1)
            # argsort两次得到排名 (0-based), +1 得到 1-based
            # 对 -fan_votes 排序 = 降序
            fan_ranks = rankdata(-fan_votes_mean, method='min')
            
            # 2. 计算总Rank
            total_score = judge_metrics + fan_ranks
            # 在Rank制下，分数越高越差
            # 排序：分数越大 -> 排名越靠后
            final_metric = total_score 
            
        else: # percent
            # 1. 观众百分比 (0-100)
            fan_pct = fan_votes_mean * 100
            # 2. 总分
            total_score = judge_metrics + fan_pct
            # 在Percent制下，分数越低越差
            # 为了统一逻辑，我们取负号，变成“数值越大越差”
            final_metric = -total_score 
            
        # --- 检查一致性 ---
        
        # 找出历史上的淘汰者
        true_eliminated = elim_rows['contestant'].tolist()
        
        # 找出模型重演后的“倒数第一” (数值最大)
        # 注意：如果有并列倒数第一，只要历史淘汰者在其中就算一致
        max_score = np.max(final_metric)
        model_bottoms = contestants[final_metric == max_score]
        
        # S28+: 裁判救人规则检查
        is_consistent = False
        consistency_note = ""
        
        if not has_save:
            # 严格规则：淘汰者必须是全场分最差的
            # 检查 true_eliminated 是否在 model_bottoms 中
            match = set(true_eliminated) & set(model_bottoms)
            if match:
                is_consistent = True
                consistency_note = "Exact Match"
            else:
                # 能够计算“排名偏差”：历史淘汰者在模型里的排名是多少？
                # 比如他实际上排倒数第3，说明模型估算有偏差
                pass
        else:
            # 宽松规则：淘汰者必须在 Bottom Two (倒数两名)
            # 计算每个人的排名 (数值越大越差)
            # 倒数第1和倒数第2的 Score 是多少
            unique_scores = np.unique(final_metric)
            sorted_scores = np.sort(unique_scores)[::-1] # 降序: 最差, 次差...
            
            bottom_threshold = sorted_scores[0] # 倒数第一的分数
            if len(sorted_scores) > 1:
                # 如果有倒数第二档分数
                # 注意：如果倒数第一有2人，则Bottom Two就是这2人，不需要取第二档
                num_at_bottom = np.sum(final_metric == sorted_scores[0])
                if num_at_bottom < 2:
                    bottom_threshold = sorted_scores[1] # 纳入倒数第二档
            
            # Bottom Two 名单
            bottom_two_candidates = contestants[final_metric >= bottom_threshold]
            
            match = set(true_eliminated) & set(bottom_two_candidates)
            if match:
                is_consistent = True
                consistency_note = "In Bottom Two (Saved)"
        
        if is_consistent:
            consistent_weeks += 1
            
        validation_results.append({
            'Season': season,
            'Week': week,
            'True_Eliminated': ", ".join(true_eliminated),
            'Model_Predicted_Risk': ", ".join(model_bottoms) if not has_save else ", ".join(bottom_two_candidates),
            'Consistent': is_consistent,
            'Note': consistency_note
        })
        
    # 计算总体一致性指标
    accuracy = consistent_weeks / total_weeks if total_weeks > 0 else 0
    print(f"一致性检验完成。总体准确率: {accuracy:.2%} ({consistent_weeks}/{total_weeks})")
    
    # 保存报告
    pd.DataFrame(validation_results).to_csv(report_file, index=False)
    print(f"详细报告已保存至 {report_file}")

if __name__ == "__main__":
    # 需要先运行前两个文件生成数据
    validate_consistency('Final_Fan_Votes_Estimated.csv', 
                         'dwts_step1_processed.csv', 
                         'Validation_Report.csv')