import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(base_dir, "task4_simulation_results.csv")
    redemption_path = os.path.join(base_dir, "task4_redemption_log.csv")
    task2_path = os.path.join(base_dir, "task2_week_summary.csv")
    
    # Load data
    sim_df = pd.read_csv(sim_path)
    red_df = pd.read_csv(redemption_path)
    hist_df = pd.read_csv(task2_path) # Contains actual_eliminated
    
    # --- 1. Fairness Check: Bobby Bones (Season 27) ---
    print("--- Case Study: Bobby Bones (Season 27) ---")
    bobby = sim_df[(sim_df['season'] == 27) & (sim_df['contestant'] == 'Bobby Bones')]
    if not bobby.empty:
        # Check if eliminated
        eliminated_row = bobby[bobby['is_eliminated'] == True]
        if not eliminated_row.empty:
            elim_week = eliminated_row.iloc[0]['week']
            stage = eliminated_row.iloc[0]['stage']
            print(f"Result: Eliminated in Week {elim_week} ({stage})")
        else:
            # Did he win?
            final_row = bobby[bobby['week'] == bobby['week'].max()]
            rank = final_row.iloc[0].get('rank', -1)
            print(f"Result: Reached Finale (Rank {rank})")
            
            # Check if he was redeemed?
            if final_row.iloc[0].get('is_redeemed'):
                 print("(via Redemption)")
    else:
        print("Data for Bobby Bones not found in simulation.")

    # --- 2. Excitement Check: Redemption Success Rate ---
    print("\n--- Redemption Success Analysis ---")
    total_redemptions = len(red_df)
    
    # Check how many redeemed contestants reached Top 3 in Finale
    success_count = 0
    champion_count = 0
    
    for idx, row in red_df.iterrows():
        s = row['season']
        winner = row['winner']
        
        # Get finale result
        finale = sim_df[(sim_df['season'] == s) & (sim_df['contestant'] == winner) & (sim_df['stage'].str.contains("Finale"))]
        if not finale.empty:
            rank = finale.iloc[0]['rank']
            if rank <= 3:
                success_count += 1
                print(f"Season {s}: {winner} redeemed -> Rank {rank}")
            if rank == 1:
                champion_count += 1
                print(f"  *** CHAMPION ***")
                
    print(f"\nTotal Redemptions: {total_redemptions}")
    print(f"Reached Top 3: {success_count} ({success_count/total_redemptions*100:.1f}%)")
    print(f"Won Championship: {champion_count} ({champion_count/total_redemptions*100:.1f}%)")
    
    # --- 3. Consistency with History (Did we keep the 'Good' Champions?) ---
    # Compare Simulated Champion vs Actual Champion
    print("\n--- Champion Comparison (Selected Seasons) ---")
    # Get actual champions (where actual_eliminated is NaN in the last week? No, data structure is tricky)
    # Actually task2_week_summary.csv has 'actual_eliminated'. The winner is the one NEVER eliminated.
    
    # Let's infer actual winner from raw data if possible, or just skip full comparison
    # We can check specific known winners like Bindi Irwin (Season 21) or Jordan Fisher (Season 25)
    
    notable_winners = {
        21: "Bindi Irwin",
        25: "Jordan Fisher",
        19: "Alfonso Ribeiro",
        31: "Charli D'Amelio"
    }
    
    for s, name in notable_winners.items():
        sim_winner = sim_df[(sim_df['season'] == s) & (sim_df['stage'].str.contains("Finale")) & (sim_df['rank'] == 1)]
        if not sim_winner.empty:
            sim_name = sim_winner.iloc[0]['contestant']
            match = "MATCH" if sim_name == name else f"DIFF ({sim_name})"
            print(f"Season {s}: Actual={name} | Sim={match}")

if __name__ == "__main__":
    main()
