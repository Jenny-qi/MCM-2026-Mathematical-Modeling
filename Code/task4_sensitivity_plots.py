
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300

def plot_heatmap(df, output_dir):
    """
    Generate Heatmap: Top-3 Rate vs Strict Weight & Penalty
    (Changed from Champion Rate to Top-3 Rate as Champion Rate is often 0 with penalty)
    """
    pivot_table = df.pivot(index='strict_judge_weight', 
                          columns='redemption_penalty', 
                          values='top3_rate')
    
    # Fix floating point issues in index and columns
    pivot_table.index = pivot_table.index.map(lambda x: round(x, 2))
    pivot_table.columns = pivot_table.columns.map(lambda x: round(x, 2))
    
    # Sort index descending for better visualization (High weight at top)
    pivot_table = pivot_table.sort_index(ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", 
                cbar_kws={'label': 'Redemption Top-3 Rate'})
    
    plt.title('Sensitivity Heatmap: Redemption Top-3 Rate', fontsize=14, pad=20)
    plt.xlabel('Redemption Penalty (Sigma)', fontsize=12)
    plt.ylabel('Strict Phase Judge Weight', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task4_sensitivity_heatmap.png'), dpi=300)
    plt.close()

def plot_bobby_curve(df, output_dir):
    """
    Generate Line Plot: Bobby Bones Elimination Week vs Strict Weight
    (Aggregating over penalties, though penalty shouldn't affect Strict Phase elimination 
     unless he is redeemed, but Bobby is usually eliminated IN Strict/Growth phase or early Finale)
    Actually, let's fix Penalty to Baseline (0.5) and vary Weight.
    """
    # Filter for baseline penalty = 0.5
    baseline_penalty_df = df[np.isclose(df['redemption_penalty'], 0.5)]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=baseline_penalty_df, x='strict_judge_weight', y='bobby_elim_week', 
                 marker='o', linewidth=2.5, color='#E74C3C')
    
    # Add threshold line for Finale (Week 8 in S27)
    plt.axhline(y=8, color='gray', linestyle='--', label='Finale Phase Start (Week 8)')
    plt.axvline(x=0.7, color='green', linestyle=':', label='Current Baseline (0.7)')
    
    plt.title('Impact of Strict Phase Judge Weight on Bobby Bones (S27)', fontsize=14, pad=20)
    plt.xlabel('Strict Phase Judge Weight', fontsize=12)
    plt.ylabel('Elimination Week', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task4_sensitivity_bobby_curve.png'), dpi=300)
    plt.close()

def plot_3d_surface(df, output_dir):
    """
    Generate 3D Surface Plot: Top-3 Rate vs Weight & Penalty
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare grid
    X = df['strict_judge_weight'].unique()
    Y = df['redemption_penalty'].unique()
    X, Y = np.meshgrid(X, Y)
    
    # Z values need to be mapped to the meshgrid
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            w = X[i, j]
            p = Y[i, j]
            # Find value
            val = df[(np.isclose(df['strict_judge_weight'], w)) & 
                     (np.isclose(df['redemption_penalty'], p))]['top3_rate'].values[0]
            Z[i, j] = val
            
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('Strict Judge Weight', fontsize=11, labelpad=10)
    ax.set_ylabel('Redemption Penalty', fontsize=11, labelpad=10)
    ax.set_zlabel('Top-3 Rate', fontsize=11, labelpad=10)
    ax.set_title('3D Surface: Redemption Top-3 Rate Sensitivity', fontsize=14, pad=20)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Top-3 Rate')
    
    # Rotate for better view
    ax.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task4_sensitivity_3d_surface.png'), dpi=300)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "task4_sensitivity_results.csv")
    output_dir = os.path.join(base_dir, "task4_outputs") # Reuse existing output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(input_path)
    
    set_style()
    
    print("Generating Heatmap...")
    plot_heatmap(df, output_dir)
    
    print("Generating Bobby Bones Curve...")
    plot_bobby_curve(df, output_dir)
    
    print("Generating 3D Surface Plot...")
    plot_3d_surface(df, output_dir)
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
