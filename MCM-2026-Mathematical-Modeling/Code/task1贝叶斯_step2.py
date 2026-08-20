import pandas as pd
import numpy as np
from tqdm import tqdm

class DWTSSolver:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.results = []
    
    def simulate_week(self, meta, contestants, judge_metrics, elim_idx, n_samples=50000):
        n_c = len(contestants)
        # 1. 先验采样 (Dirichlet): 生成得票率 %
        prior_samples = np.random.dirichlet(np.ones(n_c), n_samples)
        
        valid_mask = None
        
        # 2. 约束检查
        if meta['system'] == 'rank':
            j_ranks = np.array(judge_metrics)
            # 关键：将百分比转化为排名 (Percent -> Rank)
            # 票数越高(大) -> Rank数值越小(1)。
            f_ranks = np.argsort(np.argsort(-prior_samples, axis=1), axis=1) + 1
            total_ranks = j_ranks + f_ranks
            
            # 淘汰者的总Rank
            elim_rank_vals = total_ranks[:, elim_idx][:, None]
            
            if not meta['has_save']:
                # S1-S2: 淘汰者必须是总Rank数值最大的 (最差)
                valid_mask = (total_ranks[:, elim_idx] == np.max(total_ranks, axis=1))
            else:
                # S28+: 淘汰者必须在 Bottom Two (只有0或1人比他更差)
                worse_count = np.sum(total_ranks > elim_rank_vals, axis=1)
                valid_mask = (worse_count <= 1)
                
        else: # percent system
            j_pct = np.array(judge_metrics)
            total_score = j_pct + (prior_samples * 100)
            # S3-S27: 淘汰者必须是总分最低的
            valid_mask = (total_score[:, elim_idx] == np.min(total_score, axis=1))
            
        return prior_samples[valid_mask]

    def solve(self, output_file):
        print("[2/3] 开始运行贝叶斯反演模型...")
        # 只处理有淘汰发生的周
        groups = self.df.groupby(['season', 'week'])
        
        for (season, week), group in tqdm(groups):
            elim_rows = group[group['is_eliminated'] == True]
            if elim_rows.empty: continue # 非淘汰周跳过，留给Step 3插值
            
            elim_name = elim_rows.iloc[0]['contestant']
            contestants = group['contestant'].tolist()
            judge_metrics = group['judge_metric'].tolist()
            
            meta = {
                'system': group.iloc[0]['system'],
                'has_save': group.iloc[0]['has_save']
            }
            
            try:
                elim_idx = contestants.index(elim_name)
            except: continue
                
            posterior = self.simulate_week(meta, contestants, judge_metrics, elim_idx)
            
            if len(posterior) == 0: continue # 无解情况
            
            # 统计结果
            means = np.mean(posterior, axis=0)
            stds = np.std(posterior, axis=0)
            ci_low = np.percentile(posterior, 2.5, axis=0)
            ci_high = np.percentile(posterior, 97.5, axis=0)
            
            for i, name in enumerate(contestants):
                self.results.append({
                    'Season': season, 'Week': week, 'Contestant': name,
                    'Fan_Vote_Mean': means[i], 'Fan_Vote_Std': stds[i],
                    'CI_Lower_95': ci_low[i], 'CI_Upper_95': ci_high[i],
                    'Method': 'Bayesian' # 标记来源
                })
        
        pd.DataFrame(self.results).to_csv(output_file, index=False)
        print(f"模型计算完成 -> {output_file}")

if __name__ == "__main__":
    solver = DWTSSolver('dwts_step1_processed.csv')
    solver.solve('dwts_step2_model_results.csv')