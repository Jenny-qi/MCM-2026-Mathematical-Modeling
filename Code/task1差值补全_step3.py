import pandas as pd
import numpy as np

def fill_missing_weeks(model_res_file, raw_processed_file, final_output):
    print("[3/3] 正在进行非淘汰周的插值补全...")
    model_df = pd.read_csv(model_res_file)
    full_structure = pd.read_csv(raw_processed_file)
    
    final_rows = []
    
    # 按赛季和选手遍历，确保每个人的每场比赛都有数据
    for season in full_structure['season'].unique():
        s_df = full_structure[full_structure['season'] == season]
        
        for contestant in s_df['contestant'].unique():
            # 获取该选手所有参赛记录
            c_weeks = s_df[s_df['contestant'] == contestant]['week'].unique()
            c_weeks = sorted(c_weeks)
            
            # 获取模型已有的预测
            existing = model_df[(model_df['Season'] == season) & 
                                (model_df['Contestant'] == contestant)]
            existing_map = existing.set_index('Week').to_dict('index')
            
            for i, week in enumerate(c_weeks):
                if week in existing_map:
                    # 使用模型数据
                    row = existing_map[week]
                    row['Season'] = season
                    row['Week'] = week
                    row['Contestant'] = contestant
                    final_rows.append(row)
                else:
                    # 缺失：插值
                    # 寻找最近的前后值
                    prev_val, next_val = None, None
                    
                    # 向前搜索
                    for w in reversed(c_weeks[:i]):
                        if w in existing_map:
                            prev_val = existing_map[w]['Fan_Vote_Mean']
                            break
                    # 向后搜索
                    for w in c_weeks[i+1:]:
                        if w in existing_map:
                            next_val = existing_map[w]['Fan_Vote_Mean']
                            break
                    
                    # 决策
                    val = 0.05 # 默认兜底值
                    if prev_val is not None and next_val is not None:
                        val = (prev_val + next_val) / 2
                    elif prev_val is not None:
                        val = prev_val
                    elif next_val is not None:
                        val = next_val
                    else:
                        # 全季无数据（极罕见），给平均值
                        val = 1.0 / len(s_df[s_df['week']==week])
                    
                    final_rows.append({
                        'Season': season, 'Week': week, 'Contestant': contestant,
                        'Fan_Vote_Mean': val,
                        'Fan_Vote_Std': 0.1, # 插值的不确定性较高
                        'CI_Lower_95': max(0, val - 0.1),
                        'CI_Upper_95': min(1, val + 0.1),
                        'Method': 'Interpolation'
                    })

    final_df = pd.DataFrame(final_rows)
    # 格式化
    cols = ['Fan_Vote_Mean', 'Fan_Vote_Std', 'CI_Lower_95', 'CI_Upper_95']
    for c in cols: final_df[c] = final_df[c].round(4)
    
    final_df = final_df.sort_values(['Season', 'Week', 'Contestant'])
    final_df.to_csv(final_output, index=False)
    print(f"全部任务完成！最终文件 -> {final_output}")

if __name__ == "__main__":
    fill_missing_weeks('dwts_step2_model_results.csv', 
                       'dwts_step1_processed.csv', 
                       'Final_Fan_Votes_Estimated.csv')