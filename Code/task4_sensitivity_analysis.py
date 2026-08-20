
import pandas as pd
import numpy as np
import os
import time

def run_simulation(features_path, prep_path, strict_judge_weight, redemption_penalty):
    """
    Independent simulation engine for Task 4 Sensitivity Analysis.
    
    Args:
        features_path (str): Path to task3_features.csv
        prep_path (str): Path to task2_prepared.csv
        strict_judge_weight (float): Judge weight for Strict Phase (w_J). 
                                     Fan weight will be (1 - w_J).
        redemption_penalty (float): Z-score penalty for redeemed contestants in Finale Phase.
    
    Returns:
        dict: Performance metrics for this parameter set.
              {
                  'bobby_bones_elim_week': int,
                  'redemption_champion_rate': float,
                  'redemption_top3_rate': float
              }
    """
    
    # --- Load Data (Read-Only) ---
    df = pd.read_csv(features_path)
    df_prep = pd.read_csv(prep_path)
    
    seasons = sorted(df['season'].unique())
    redemption_log = []
    simulation_results = []
    
    for s in seasons:
        season_data = df[df['season'] == s].copy()
        weeks = sorted(season_data['week'].unique())
        final_week = max(weeks)
        
        # --- Dynamic Phase Logic (Copied & Parameterized) ---
        finale_weeks = [final_week - 2, final_week - 1, final_week]
        finale_weeks = [w for w in finale_weeks if w > 0]
        
        redemption_week = final_week - 3
        if redemption_week < 1: redemption_week = 0
        
        if redemption_week > 1:
            mid_point = redemption_week // 2
        else:
            mid_point = 0
            
        contestants = set(season_data[season_data['week'] == weeks[0]]['contestant'].unique())
        eliminated_pool = {} 
        active_contestants = list(contestants)
        current_active = set(active_contestants)
        
        contestant_perf = season_data.groupby('contestant')[['judge_z', 'fan_z']].mean().to_dict('index')
        
        for w in weeks:
            is_redemption_now = (w == redemption_week)
            is_finale_phase = (w in finale_weeks)
            
            # --- Apply Parameter: Strict Phase Weight ---
            if is_finale_phase:
                w_j, w_f = 0.3, 0.7
                stage = "Finale Phase"
            elif is_redemption_now:
                w_j, w_f = 0.5, 0.5
                stage = "Redemption Week"
            elif w <= mid_point:
                # PARAMETER APPLIED HERE
                w_j = strict_judge_weight
                w_f = 1.0 - strict_judge_weight
                stage = "Phase 1 (Strict)"
            else:
                w_j, w_f = 0.5, 0.5
                stage = "Phase 2 (Growth)"
                
            # --- Calculate Scores ---
            week_df = season_data[season_data['week'] == w].set_index('contestant')
            week_scores = []
            
            for c in list(current_active):
                if c in week_df.index:
                    jz = week_df.loc[c, 'judge_z']
                    fz = week_df.loc[c, 'fan_z']
                    if isinstance(jz, pd.Series): jz = jz.mean()
                    if isinstance(fz, pd.Series): fz = fz.mean()
                else:
                    jz = contestant_perf.get(c, {'judge_z': 0})['judge_z']
                    fz = contestant_perf.get(c, {'fan_z': 0})['fan_z']
                
                # --- Apply Parameter: Redemption Penalty ---
                handicap = 0
                is_redeemed = False
                for r in redemption_log:
                    if r['season'] == s and r['winner'] == c:
                        is_redeemed = True
                        if is_finale_phase: 
                            handicap = -1 * abs(redemption_penalty) # Ensure negative
                        break

                score = w_j * jz + w_f * fz + handicap
                
                week_scores.append({
                    'season': s, 'week': w, 'contestant': c,
                    'total_score': score,
                    'is_redeemed': is_redeemed,
                    'is_eliminated': False,
                    'rank': 0
                })
            
            if not week_scores: continue
            ws_df = pd.DataFrame(week_scores)
            
            # --- Elimination Logic ---
            ws_df = ws_df.sort_values('total_score', ascending=True)
            
            # Rank Assignment (1 is highest score)
            # sort is ascending, so last row is rank 1
            for idx, row in ws_df.iterrows():
                ws_df.at[idx, 'rank'] = len(ws_df) - list(ws_df.index).index(idx)

            hist_elim_count = df_prep[(df_prep['season']==s) & (df_prep['week']==w) & (df_prep['is_eliminated']==True)].shape[0]
            if hist_elim_count == 0: hist_elim_count = 1
            
            if is_finale_phase and len(current_active) <= 3 and w != final_week:
                hist_elim_count = 0
            
            eliminated_this_week = []
            if hist_elim_count > 0 and len(current_active) > 1:
                eliminated_this_week = ws_df.head(hist_elim_count)['contestant'].tolist()
                
                for elim in eliminated_this_week:
                    current_active.remove(elim)
                    if not is_finale_phase and not is_redemption_now:
                        avg_j = contestant_perf[elim]['judge_z']
                        avg_f = contestant_perf[elim]['fan_z']
                        comp_score = 0.5 * avg_j + 0.5 * avg_f
                        eliminated_pool[elim] = comp_score

            # Update Simulation Results with Status
            for idx, row in ws_df.iterrows():
                contestant = row['contestant']
                is_elim = (contestant in eliminated_this_week)
                # Correction for final week winner
                if w == final_week and row['rank'] == 1:
                    is_elim = False
                
                # We need to store this for metrics calculation
                simulation_results.append({
                    'season': s, 'week': w, 'contestant': contestant,
                    'rank': row['rank'],
                    'is_redeemed': row['is_redeemed'],
                    'is_eliminated': is_elim
                })

            # --- Process Redemption ---
            if is_redemption_now and len(eliminated_pool) > 0:
                sorted_redemption = sorted(eliminated_pool.items(), key=lambda x: x[1], reverse=True)
                top_3_candidates = sorted_redemption[:3]
                candidates_names = [x[0] for x in top_3_candidates]
                
                winner = None
                highest_fan_z = -999
                for cand in candidates_names:
                    fz = contestant_perf[cand]['fan_z']
                    if fz > highest_fan_z:
                        highest_fan_z = fz
                        winner = cand
                
                if winner:
                    current_active.add(winner)
                    redemption_log.append({
                        'season': s, 'week': w, 
                        'winner': winner
                    })

    # --- Calculate Metrics ---
    sim_df = pd.DataFrame(simulation_results)
    
    # Metric 1: Bobby Bones (S27) Elimination Week
    bobby = sim_df[(sim_df['season'] == 27) & (sim_df['contestant'] == 'Bobby Bones')]
    if not bobby.empty:
        # Find last active week
        # If never eliminated, he is winner
        eliminated_rows = bobby[bobby['is_eliminated'] == True]
        if not eliminated_rows.empty:
            bobby_elim_week = eliminated_rows['week'].min()
        else:
            # Check if he won (rank 1 in final week)
            last_week_row = bobby[bobby['week'] == bobby['week'].max()]
            if not last_week_row.empty and last_week_row.iloc[0]['rank'] == 1:
                bobby_elim_week = 99 # Code for Winner
            else:
                bobby_elim_week = bobby['week'].max() # Survived until end but didn't win
    else:
        bobby_elim_week = 0 # Not found

    # Metric 2 & 3: Redemption Performance
    # Filter for redeemed contestants
    redeemed_seasons = [r['season'] for r in redemption_log]
    redeemed_winners = [r['winner'] for r in redemption_log]
    
    redeemed_final_ranks = []
    
    for s, w in zip(redeemed_seasons, redeemed_winners):
        # Find their final rank in that season
        rows = sim_df[(sim_df['season'] == s) & (sim_df['contestant'] == w)]
        if not rows.empty:
            # Their last appearance rank
            last_rank = rows.sort_values('week', ascending=False).iloc[0]['rank']
            redeemed_final_ranks.append(last_rank)
            
    total_redeemed = len(redeemed_final_ranks)
    if total_redeemed > 0:
        champion_count = sum([1 for r in redeemed_final_ranks if r == 1])
        top3_count = sum([1 for r in redeemed_final_ranks if r <= 3])
        champion_rate = champion_count / total_redeemed
        top3_rate = top3_count / total_redeemed
    else:
        champion_rate = 0.0
        top3_rate = 0.0
        
    return {
        'bobby_bones_elim_week': bobby_elim_week,
        'redemption_champion_rate': champion_rate,
        'redemption_top3_rate': top3_rate,
        'total_redeemed_count': total_redeemed
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_dir, "task3_features.csv")
    prep_path = os.path.join(base_dir, "task2_prepared.csv")
    output_path = os.path.join(base_dir, "task4_sensitivity_results.csv")
    
    # --- Define Parameter Ranges ---
    # Strict Weight: 0.4 to 0.9 (Baseline 0.7)
    strict_weights = np.linspace(0.4, 0.9, 11) # 0.4, 0.45, ... 0.9
    
    # Penalty: 0.0 to 1.0 (Baseline 0.5)
    penalties = np.linspace(0.0, 1.0, 11) # 0.0, 0.1, ... 1.0
    
    print(f"Starting Sensitivity Analysis...")
    print(f"Scanning {len(strict_weights)} weights x {len(penalties)} penalties = {len(strict_weights)*len(penalties)} combinations.")
    
    results = []
    
    start_time = time.time()
    counter = 0
    total = len(strict_weights) * len(penalties)
    
    for w_j in strict_weights:
        for p in penalties:
            counter += 1
            if counter % 10 == 0:
                print(f"Processing {counter}/{total}...")
                
            metrics = run_simulation(features_path, prep_path, w_j, p)
            
            results.append({
                'strict_judge_weight': w_j,
                'redemption_penalty': p,
                'bobby_elim_week': metrics['bobby_bones_elim_week'],
                'champion_rate': metrics['redemption_champion_rate'],
                'top3_rate': metrics['redemption_top3_rate'],
                'sample_size': metrics['total_redeemed_count']
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    
    print(f"Analysis Complete. Results saved to {output_path}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
