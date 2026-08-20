import pandas as pd
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Load data
    features_path = os.path.join(base_dir, "task3_features.csv")
    df = pd.read_csv(features_path)

    # Load prep data for historical elimination count
    prep_path = os.path.join(base_dir, "task2_prepared.csv")
    df_prep = pd.read_csv(prep_path)
    
    # --- Configuration Parameters ---
    # Can be adjusted for sensitivity analysis or tuning
    STRICT_PHASE_JUDGE_WEIGHT = 0.7  # Default: 0.7 (70%)
    REDEMPTION_PENALTY_SIGMA = 0.3   # Default: 0.5 (Z-Score)
    
    simulation_results = []
    redemption_log = []
    
    seasons = sorted(df['season'].unique())
    
    for s in seasons:
        season_data = df[df['season'] == s].copy()
        weeks = sorted(season_data['week'].unique())
        final_week = max(weeks)
        
        # --- Define Dynamic Phases based on User's New Logic ---
        # 1. Finale Phase = Last 3 weeks (Final-2, Final-1, Final)
        finale_weeks = [final_week - 2, final_week - 1, final_week]
        finale_weeks = [w for w in finale_weeks if w > 0] # Safety check
        
        # 2. Redemption Night = The week BEFORE Finale Phase
        # i.e., Final-3 (倒数第4周)
        redemption_week = final_week - 3
        if redemption_week < 1: redemption_week = 0 # Handle very short seasons
        
        # 3. Regular Phases
        # Strict Phase (Phase 1): Week 1 to Middle
        # Growth Phase (Phase 2): Middle to Redemption
        # Let's split remaining weeks evenly
        if redemption_week > 1:
            mid_point = redemption_week // 2
        else:
            mid_point = 0
            
        # Track active contestants
        contestants = set(season_data[season_data['week'] == weeks[0]]['contestant'].unique())
        eliminated_pool = {} # contestant -> average_score (for redemption)
        
        active_contestants = list(contestants)
        
        # Pre-calculate average performance for all contestants in this season
        # Used for imputation and redemption eligibility
        contestant_perf = season_data.groupby('contestant')[['judge_z', 'fan_z']].mean().to_dict('index')
        
        current_active = set(active_contestants)
        
        # Simulate week by week
        for w in weeks:
            # --- Determine Stage & Weights ---
            is_redemption_now = (w == redemption_week)
            is_finale_phase = (w in finale_weeks)
            
            if is_finale_phase:
                w_j, w_f = 0.3, 0.7
                stage = "Finale Phase"
            elif is_redemption_now:
                # Redemption week itself is special (see logic below)
                # But regular contestants still compete? 
                # Usually Redemption Night is a special event. 
                # Let's assume Regulars compete normally (Growth weights), AND Redemption happens in parallel.
                w_j, w_f = 0.5, 0.5
                stage = "Redemption Week"
            elif w <= mid_point:
                w_j, w_f = STRICT_PHASE_JUDGE_WEIGHT, 1.0 - STRICT_PHASE_JUDGE_WEIGHT
                stage = "Phase 1 (Strict)"
            else:
                w_j, w_f = 0.5, 0.5
                stage = "Phase 2 (Growth)"
                
            # --- Calculate Scores ---
            week_df = season_data[season_data['week'] == w].set_index('contestant')
            week_scores = []
            
            for c in list(current_active):
                # Get performance (Impute if missing)
                if c in week_df.index:
                    jz = week_df.loc[c, 'judge_z']
                    fz = week_df.loc[c, 'fan_z']
                    if isinstance(jz, pd.Series): jz = jz.mean()
                    if isinstance(fz, pd.Series): fz = fz.mean()
                else:
                    jz = contestant_perf.get(c, {'judge_z': 0})['judge_z']
                    fz = contestant_perf.get(c, {'fan_z': 0})['fan_z']
                
                # Handicap for redeemed in Finale Phase?
                handicap = 0
                is_redeemed = False
                for r in redemption_log:
                    if r['season'] == s and r['winner'] == c:
                        is_redeemed = True
                        if is_finale_phase: handicap = -1 * abs(REDEMPTION_PENALTY_SIGMA) # Penalty in finale
                        break

                score = w_j * jz + w_f * fz + handicap
                
                week_scores.append({
                    'season': s, 'week': w, 'contestant': c,
                    'judge_z': jz, 'fan_z': fz,
                    'weight_j': w_j, 'weight_f': w_f,
                    'total_score': score,
                    'stage': stage,
                    'status': 'active',
                    'is_redeemed': is_redeemed
                })
            
            if not week_scores: continue
            ws_df = pd.DataFrame(week_scores)
            
            # --- Elimination Logic ---
            ws_df = ws_df.sort_values('total_score', ascending=True)
            
            # How many to eliminate?
            # Finale Phase: Usually keeps 3-4 people.
            # We follow historical pace mostly.
            hist_elim_count = df_prep[(df_prep['season']==s) & (df_prep['week']==w) & (df_prep['is_eliminated']==True)].shape[0]
            if hist_elim_count == 0: hist_elim_count = 1 
            
            # Special: Don't eliminate if only 3 people left in Finale Phase (just ranking)
            # Unless it's the very last week (Champion decided)
            if is_finale_phase and len(current_active) <= 3 and w != final_week:
                hist_elim_count = 0
            
            eliminated_this_week = []
            if hist_elim_count > 0 and len(current_active) > 1: # Always keep 1 winner
                eliminated_this_week = ws_df.head(hist_elim_count)['contestant'].tolist()
                
                for elim in eliminated_this_week:
                    current_active.remove(elim)
                    # Add to pool for redemption (if redemption hasn't passed)
                    if not is_finale_phase and not is_redemption_now:
                        # Eligibility Score: Avg(0.5*J + 0.5*F)
                        # "历史平均综合得分"
                        avg_j = contestant_perf[elim]['judge_z']
                        avg_f = contestant_perf[elim]['fan_z']
                        comp_score = 0.5 * avg_j + 0.5 * avg_f
                        eliminated_pool[elim] = comp_score

            # --- Process Redemption Night Logic ---
            # Trigger: At the END of 'Redemption Week' (before Finale Phase starts)
            if is_redemption_now and len(eliminated_pool) > 0:
                # Rank by Historical Average Composite Score
                sorted_redemption = sorted(eliminated_pool.items(), key=lambda x: x[1], reverse=True)
                top_3_candidates = sorted_redemption[:3]
                
                # Winner: Highest Fan_Z among Top 3 (Round 2 Logic)
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
                        'season': s, 'week': w, # Redemption happened this week
                        'candidates': candidates_names,
                        'winner': winner,
                        'winner_score': eliminated_pool[winner]
                    })
                    # Add a virtual record for the winner's return
                    simulation_results.append({
                        'season': s, 'week': w, 'contestant': winner,
                        'judge_z': contestant_perf[winner]['judge_z'],
                        'fan_z': contestant_perf[winner]['fan_z'],
                        'weight_j': 0, 'weight_f': 0, 'total_score': 0,
                        'stage': 'REDEMPTION WIN',
                        'status': 'redeemed',
                        'is_eliminated': False
                    })

            # Record regular results
            for idx, row in ws_df.iterrows():
                res = row.to_dict()
                # If rank 1 in final week, they are winner, not eliminated
                if w == final_week:
                    res['is_eliminated'] = (idx < len(ws_df)-1) # Only last one (highest score) survives? 
                    # Wait, sort is ascending (lowest first). So last row is winner.
                    # idx 0 is lowest score.
                    # Winner is the one with highest score.
                else:
                    res['is_eliminated'] = (row['contestant'] in eliminated_this_week)
                
                # Add rank info
                # Calculate correct rank based on position in sorted dataframe
                # ws_df is sorted ascending by score (lowest score first)
                # So Rank = len(ws_df) - position_index
                position_index = ws_df.index.get_loc(idx)
                res['rank'] = len(ws_df) - position_index
                
                simulation_results.append(res)

    # Save Results
    sim_df = pd.DataFrame(simulation_results)
    sim_df.to_csv(os.path.join(base_dir, "task4_simulation_results.csv"), index=False)
    
    red_df = pd.DataFrame(redemption_log)
    red_df.to_csv(os.path.join(base_dir, "task4_redemption_log.csv"), index=False)
    
    print("Task 4 Refined Simulation Complete.")

if __name__ == "__main__":
    main()
