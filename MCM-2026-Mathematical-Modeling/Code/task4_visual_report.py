import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import numpy as np

def main():
    # --- Style Configuration ---
    try:
        plt.style.use('seaborn-v0_8-white')
    except:
        plt.style.use('seaborn-white')
        
    mpl.rcParams['font.family'] = 'sans-serif' # Use sans-serif to be safe, or Arial if available
    # mpl.rcParams['font.sans-serif'] = ['Arial'] # Optional
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.color'] = '#333333'
    mpl.rcParams['ytick.color'] = '#333333'
    mpl.rcParams['axes.labelcolor'] = '#333333'
    mpl.rcParams['axes.edgecolor'] = '#333333'

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(base_dir, "task4_simulation_results.csv")
    output_dir = os.path.join(base_dir, "task4_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(sim_path)
    
    # --- 1. Bobby Bones Trajectory (Season 27) ---
    # Enhanced with Dynamic Phase Visualization
    
    season_27 = df[df['season'] == 27]
    bobby = season_27[season_27['contestant'] == 'Bobby Bones']
    
    if not bobby.empty:
        # Determine phases for Season 27
        # Season 27 final week is 11 (based on data)
        # Finale Phase: 9, 10, 11
        # Redemption Night: 8
        # Strict Phase: 1 - 4
        # Growth Phase: 5 - 7
        
        final_week = season_27['week'].max()
        redemption_week = final_week - 3
        finale_start = final_week - 2
        
        weeks = sorted(season_27['week'].unique())
        
        # Collect scores
        bobby_scores = []
        cutoff_scores = []
        
        for w in weeks:
            week_data = season_27[season_27['week'] == w]
            # Bobby's score (if active)
            b_row = week_data[week_data['contestant'] == 'Bobby Bones']
            if not b_row.empty:
                b_score = b_row['total_score'].values[0]
            else:
                b_score = np.nan # Eliminated
            
            # Cutoff score (the lowest score of someone who survived this week)
            # If elimination happened, it's the score of the lowest survivor
            # If no elimination, it's just min score
            survivors = week_data[week_data['is_eliminated'] == False]
            if not survivors.empty:
                cutoff = survivors['total_score'].min()
            else:
                cutoff = week_data['total_score'].min()
                
            bobby_scores.append(b_score)
            cutoff_scores.append(cutoff)

        # ==================== Create Plot (Nature Style) ====================
        plt.figure(figsize=(6.7, 4.5), dpi=300)
        ax = plt.gca()
        
        # Plot Scores (Bobby)
        plt.plot(weeks, bobby_scores, marker='o', markersize=5, 
                 color='#2874A6', linewidth=1.5, label='Bobby Bones', 
                 zorder=5, markerfacecolor='white', markeredgewidth=1.5)
        
        # Plot Scores (Cutoff)
        plt.plot(weeks, cutoff_scores, linestyle='--', color='#95A5A6', 
                 linewidth=1, alpha=0.8, label='Survival Threshold', zorder=4)
        
        # Add Phases Background (Nature Colors)
        phase_configs = [
            (0.5, 4.5, '#E8F4F8', 'Strict\n(70% Judge)'),
            (4.5, redemption_week - 0.5, '#FFF8DC', 'Growth\n(50% Judge)'),
            (redemption_week - 0.5, redemption_week + 0.5, '#FCE4EC', 'Redemption'),
            (redemption_week + 0.5, final_week + 0.5, '#E8EAF6', 'Finale\n(30% Judge)')
        ]
        
        for xmin, xmax, color, label in phase_configs:
            plt.axvspan(xmin, xmax, facecolor=color, alpha=0.6, edgecolor='none', zorder=0)
            # Add label
            center = (xmin + xmax) / 2
            plt.text(center, -0.8, label, ha='center', va='top', fontsize=6.5, color='#666666', style='italic', zorder=1)

        # Annotate Elimination
        valid_weeks = [w for i, w in enumerate(weeks) if not np.isnan(bobby_scores[i])]
        elim_week = valid_weeks[-1] if valid_weeks else 0
        
        if elim_week < final_week:
            elim_idx = weeks.index(elim_week)
            elim_score = bobby_scores[elim_idx]
            
            # Mark elimination point
            plt.plot(elim_week, elim_score, marker='o', markersize=7, 
                     color='#C0392B', markerfacecolor='white', markeredgewidth=1.5, zorder=6)
            
            plt.annotate('Eliminated', 
                         xy=(elim_week, elim_score - 0.03),
                         xytext=(elim_week, elim_score - 0.8),
                         fontsize=7, fontweight='bold', color='#DC6457', ha='center', va='top',
                         arrowprops=dict(arrowstyle='->', color='#DC6457', lw=1, connectionstyle='arc3,rad=0'),
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#DC6457', linewidth=1, alpha=0.95),
                         zorder=7)

        plt.title('Case Study: Bobby Bones vs New System (Season 27)', fontsize=10, pad=15, color='#2C3E50')
        plt.xlabel('Competition Week', fontsize=9, fontweight='bold', labelpad=5)
        plt.ylabel('Composite Score (Z-Score)', fontsize=9, fontweight='bold', labelpad=5)
        plt.axhline(y=0, color='#BDC3C7', linestyle=':', linewidth=0.8, alpha=0.7, zorder=1)
        
        # Spines & Grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, axis='y', linestyle='-', alpha=0.2, color='#BDC3C7', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend
        plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, handlelength=1.5)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(output_dir, "task4_bobby_trajectory.png"), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # --- 2. Redemption Impact (Nature Style) ---
    redeemed = df[df['is_redeemed'] == True]
    if not redeemed.empty:
        # Get their final ranks (Best rank achieved by each redeemed contestant)
        # Group by season and contestant to ensure unique counting
        final_ranks_series = redeemed.groupby(['season', 'contestant'])['rank'].min()
        final_ranks = final_ranks_series.value_counts().sort_index()
        ranks = final_ranks.index.astype(int).tolist()
        counts = final_ranks.values.astype(int).tolist()
        
        # Calculate stats for annotation
        total_redeemed = final_ranks.sum()
        # Calculate Top-3 Rate instead of Champion Rate
        top3_count = final_ranks.get(1, 0) + final_ranks.get(2, 0) + final_ranks.get(3, 0)
        top3_rate = (top3_count / total_redeemed) * 100
        
        # Stats text box
        stats_text = f'n = {total_redeemed}\nTop-3 rate: {top3_rate:.1f}%'
        def get_rank_color(rank):
            if rank == 1:
                return '#ef764f'    # Deep Gold
            elif rank == 2:
                return '#FBB475'    # Silver
            elif rank == 3:
                return '#FFF2AD'    # Bronze
            elif rank == 4:
                return '#f4fad4'    
            elif rank == 5:
                return '#b8d7e9'    
            elif rank == 6:
                return '#7dacd1'    
            else:
                return '#7F8C8D'    # Neutral Gray
                
        bar_colors = [get_rank_color(r) for r in ranks]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(3.35, 2.8), dpi=300)
        
        bars = ax.bar(ranks, counts, color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.9, width=0.6)
        
        # Add stats annotation
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, color='#666666', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#DDDDDD', alpha=0.9, linewidth=0.5))

        # Value labels
        for bar, count, rank in zip(bars, counts, ranks):
            height = bar.get_height()
            ax.annotate(f'{count}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7, fontweight='bold',
                        color='#2C3E50' if rank <= 3 else '#555555',
                        zorder=4)
            
            # Percentage for champion
            if rank == 1:
                pct = (count / total_redeemed) * 100
                ax.annotate(f'({pct:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 12),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=6, color='#666666', style='italic')

        # Axis and Title
        ax.set_title('Final ranks of redemption winners (34 seasons simulated)', 
                     fontsize=9, fontweight='bold', color='#2C3E50', pad=12)
        ax.set_xlabel('Final rank achieved', fontsize=8, fontweight='bold', labelpad=5)
        ax.set_ylabel('Number of contestants', fontsize=8, fontweight='bold', labelpad=5)
        
        # Ticks
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks], fontsize=8)
        
        y_max = max(counts) * 1.2 if counts else 10
        ax.set_ylim(0, y_max)
        ax.set_yticks(range(0, int(y_max)+1, max(1, int(max(counts)/4))))
        
        # Grid and Spines
        ax.grid(axis='y', linestyle='-', alpha=0.15, color='#7F8C8D', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Stats Annotation
        stats_text = f'n = {total_redeemed}\nTop-3 rate: {top3_rate:.1f}%'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, color='#666666', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#DDDDDD', alpha=0.9, linewidth=0.5))
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(output_dir, "task4_redemption_ranks.png"), dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', format='png')
        # Also save PDF as in reference code? The user didn't explicitly ask but it's good practice. 
        # But the original code didn't save PDF, so maybe I should stick to PNG or just PNG is enough.
        # The user said "reference this style", so I will stick to PNG as per original file, or add PDF if useful.
        # I'll stick to PNG to be safe, or just add PDF as extra. The prompt said "Refer to ... style ... modify ... style".
        # The provided snippet saves PDF too. I'll add PDF just in case.
        plt.savefig(os.path.join(output_dir, "task4_redemption_ranks.pdf"), bbox_inches='tight', 
                    facecolor='white', format='pdf')
        plt.close()

if __name__ == "__main__":
    main()
