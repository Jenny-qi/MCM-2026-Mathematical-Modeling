import pandas as pd
import numpy as np
from scipy.stats import rankdata
import re

def preprocess_dwts_data(input_file, output_file):
    print(f"[1/3] 正在预处理数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("错误：未找到输入文件。请确保文件名正确。")
        return

    df.columns = [c.lower().strip() for c in df.columns]
    processed_rows = []
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        
        # --- 规则定义 ---
        # S1-S2: Rank, Direct Elimination
        # S3-S27: Percent, Direct Elimination
        # S28+: Rank, Judges' Save (Bottom Two)
        if season <= 2:
            system = 'rank'
            has_save = False
        elif season <= 27:
            system = 'percent'
            has_save = False
        else:
            system = 'rank'
            has_save = True
            
        week_cols = [c for c in df.columns if 'week' in c and 'score' in c]
        weeks = sorted(list(set([int(re.search(r'week(\d+)', c).group(1)) for c in week_cols if re.search(r'week(\d+)', c)])))
        
        for week in weeks:
            judge_cols = [c for c in df.columns if f'week{week}_' in c and 'score' in c]
            if not judge_cols: continue
            
            week_data = []
            for _, row in season_df.iterrows():
                contestant = row['celebrity_name']
                status = row['results']
                
                scores = []
                for col in judge_cols:
                    val = row[col]
                    if pd.isna(val) or str(val).strip().upper() == 'N/A': continue
                    try:
                        scores.append(float(val))
                    except: pass
                
                if not scores or sum(scores) == 0: continue
                
                # 判断淘汰：根据 "Eliminated Week X"
                is_eliminated = False
                if isinstance(status, str) and f"Eliminated Week {week}" in status:
                    is_eliminated = True
                
                week_data.append({
                    'season': season,
                    'week': week,
                    'contestant': contestant,
                    'judge_score_raw': sum(scores),
                    'is_eliminated': is_eliminated,
                    'system': system,
                    'has_save': has_save
                })
            
            # 计算当周裁判指标
            if len(week_data) > 1:
                raw_scores = np.array([x['judge_score_raw'] for x in week_data])
                
                if system == 'rank':
                    # Rank制：分数高 -> Rank值小(1)。使用 min 处理并列。
                    # argsort默认升序，故对负分排序
                    ranks = rankdata(-raw_scores, method='min')
                    for i, r in enumerate(ranks):
                        week_data[i]['judge_metric'] = r
                else:
                    # Percent制：分数占比 (0-100)
                    total = np.sum(raw_scores)
                    for i, s in enumerate(raw_scores):
                        week_data[i]['judge_metric'] = (s / total * 100) if total > 0 else 0
                
                processed_rows.extend(week_data)
    
    pd.DataFrame(processed_rows).to_csv(output_file, index=False)
    print(f"预处理完成 -> {output_file}")

if __name__ == "__main__":
    preprocess_dwts_data('2026_MCM_Problem_C_Data.csv', 'dwts_step1_processed.csv')