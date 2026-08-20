import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coef_path = os.path.join(base_dir, "task3_fixed_effects.csv")
    pro_path = os.path.join(base_dir, "task3_pro_effects.csv")
    out_dir = os.path.join(base_dir, "task3_outputs")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 1. 绘制固定效应对比图 (Coefficient Plot)
    if os.path.exists(coef_path):
        coef_df = pd.read_csv(coef_path)
        
        # 简化 term 名字
        def clean_term(t):
            if "Intercept" in t: return "Intercept"
            if "age_std" in t: return "Age (Std)"
            # C(industry_group, ...)[T.Athlete]
            if "industry_group" in t:
                # 提取 Athlete 等
                pattern = r"\[T\.(.*?)\]"
                match = re.search(pattern, t)
                if match: return match.group(1)
            return t
            
        coef_df["term_clean"] = coef_df["term"].apply(clean_term)
        
        # 去掉 Intercept (通常数值范围不同，且不那么重要)
        plot_df = coef_df[coef_df["term_clean"] != "Intercept"].copy()
        
        plt.figure(figsize=(10, 6))
        
        # 使用 dodge 效果并排画
        sns.pointplot(data=plot_df, x="coef", y="term_clean", hue="model", 
                      dodge=0.4, join=False, capsize=0.2,
                      errorbar=None) # 我们手动画误差棒
                      
        # 手动添加误差棒
        # 由于 seaborn pointplot 对自定义误差棒支持有限，我们直接用 matplotlib
        
        plt.clf() # 清空
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 分开 Judge 和 Fan
        judge_data = plot_df[plot_df["model"] == "Judge"].sort_values("term_clean")
        fan_data = plot_df[plot_df["model"] == "Fan"].sort_values("term_clean")
        
        y_pos = range(len(judge_data))
        offset = 0.2
        
        # 画 Judge
        ax.errorbar(judge_data["coef"], [y - offset for y in y_pos], 
                    xerr=[judge_data["coef"] - judge_data["ci_lower"], judge_data["ci_upper"] - judge_data["coef"]],
                    fmt='o', label='Judge Model', capsize=5, color='blue')
                    
        # 画 Fan
        # 确保顺序一致
        fan_data = fan_data.set_index("term_clean").reindex(judge_data["term_clean"]).reset_index()
        ax.errorbar(fan_data["coef"], [y + offset for y in y_pos], 
                    xerr=[fan_data["coef"] - fan_data["ci_lower"], fan_data["ci_upper"] - fan_data["coef"]],
                    fmt='s', label='Fan Model', capsize=5, color='orange')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(judge_data["term_clean"])
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Effect Size (Standardized)")
        ax.set_title("Impact of Factors: Judge vs Fan (with 95% CI)")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "task3_fixed_effects_comparison.png"))
        plt.close()
        print(f"Plot saved: task3_fixed_effects_comparison.png")
        
    # 2. 绘制 Pro Dancer 影响力散点图 (Judge vs Fan)
    if os.path.exists(pro_path):
        pro_df = pd.read_csv(pro_path)
        
        # 清洗名字 pro[C(pro_dancer)[Derek Hough]] -> Derek Hough
        def clean_pro_name(n):
            match = re.search(r"\[(.*?)\]$", n) # 取最后一个 [] 内容
            if match:
                inner = match.group(1)
                # 如果是 C(pro_dancer)[Derek Hough]
                if "[" in inner:
                    return inner.split("[")[-1].replace("]", "")
                return inner
            return n
            
        pro_df["name_clean"] = pro_df["name"].apply(clean_pro_name)
        
        # 过滤掉 NaN (如果有)
        pro_df = pro_df.dropna(subset=["judge_effect", "fan_effect"])
        
        plt.figure(figsize=(10, 8))
        
        # 散点图
        sns.scatterplot(data=pro_df, x="judge_effect", y="fan_effect", alpha=0.7)
        
        # 标注 Top Pro (距离原点远的，或者特定的)
        # 计算距离
        pro_df["dist"] = (pro_df["judge_effect"]**2 + pro_df["fan_effect"]**2)**0.5
        top_pros = pro_df.nlargest(8, "dist")
        
        for _, row in top_pros.iterrows():
            plt.text(row["judge_effect"]+0.01, row["fan_effect"]+0.01, 
                     row["name_clean"], fontsize=9)
            
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("Pro Impact on Judge Score (BLUP)")
        plt.ylabel("Pro Impact on Fan Vote (BLUP)")
        plt.title("Pro Dancer Capability Map: Technical vs Popularity")
        
        # 添加象限说明
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.text(xlim[1]*0.8, ylim[1]*0.8, "Double Strong", color='green', ha='center')
        plt.text(xlim[1]*0.8, ylim[0]*0.8, "Judge Favored", color='blue', ha='center')
        plt.text(xlim[0]*0.8, ylim[1]*0.8, "Fan Favored", color='orange', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "task3_pro_impact_map.png"))
        plt.close()
        print(f"Plot saved: task3_pro_impact_map.png")

if __name__ == "__main__":
    main()
