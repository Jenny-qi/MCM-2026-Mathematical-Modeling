import os
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 路径定义
    raw_data_path = os.path.join(base_dir, "2026_MCM_Problem_C_Data.csv")
    step1_path = os.path.join(base_dir, "dwts_step1_processed.csv")
    fan_path = os.path.join(base_dir, "Final_Fan_Votes_Estimated.csv")
    output_path = os.path.join(base_dir, "task3_features.csv")
    
    # 2. 读取数据
    print(f"Loading data from {raw_data_path}...")
    try:
        raw_df = pd.read_csv(raw_data_path)
        step1_df = pd.read_csv(step1_path)
        fan_df = pd.read_csv(fan_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. 规范化列名
    raw_df.columns = [c.lower().strip() for c in raw_df.columns]
    step1_df.columns = [c.lower().strip() for c in step1_df.columns]
    fan_df.columns = [c.lower().strip() for c in fan_df.columns]
    
    # Fan数据重命名以匹配合并键
    fan_df = fan_df.rename(columns={
        "season": "season",
        "week": "week",
        "contestant": "contestant",
        "fan_vote_mean": "fan_vote_mean",
        "fan_vote_std": "fan_vote_std",
        "method": "fan_method"
    })

    # 4. 合并数据
    # 先合并 Step1 (Judge Score) 与 Fan (Fan Vote)
    # 以 season, week, contestant 为键
    print("Merging Judge Scores and Fan Votes...")
    merged = pd.merge(step1_df, fan_df, on=["season", "week", "contestant"], how="left")
    
    # 再合并选手静态特征 (Age, Industry, Pro Partner)
    # 从原始数据中提取 contestant -> feature 的映射表
    # 注意：原始数据是宽表，每行是一个选手在一个赛季的数据，但也包含每周评委分
    # 我们只需要提取 contestant 维度的静态信息
    
    # 提取选手特征表 (去重，每个选手每个赛季一条)
    feature_cols = [
        "season", "celebrity_name", "ballroom_partner", 
        "celebrity_industry", "celebrity_age_during_season"
    ]
    # 检查列名是否存在 (不同版本csv可能略有差异)
    existing_cols = [c for c in feature_cols if c in raw_df.columns]
    
    contestant_features = raw_df[existing_cols].drop_duplicates(subset=["season", "celebrity_name"])
    contestant_features = contestant_features.rename(columns={
        "celebrity_name": "contestant",
        "ballroom_partner": "pro_dancer",
        "celebrity_industry": "industry",
        "celebrity_age_during_season": "age"
    })
    
    print("Merging Contestant Features...")
    full_df = pd.merge(merged, contestant_features, on=["season", "contestant"], how="left")
    
    # 5. 特征工程
    
    # 5.1 处理 Age (转数值，处理缺失)
    # 检查 age 是否有非数值字符
    full_df["age"] = pd.to_numeric(full_df["age"], errors="coerce")
    # 填充缺失年龄 (用总体中位数)
    if full_df["age"].isna().sum() > 0:
        median_age = full_df["age"].median()
        full_df["age"] = full_df["age"].fillna(median_age)
        print(f"Filled missing ages with median: {median_age}")
        
    # 年龄标准化 (Z-score)
    age_mean = full_df["age"].mean()
    age_std = full_df["age"].std()
    full_df["age_std"] = (full_df["age"] - age_mean) / age_std
    
    # 5.2 处理 Industry (清洗与归类)
    # 查看 Top industries
    # print(full_df["industry"].value_counts().head(10))
    
    def clean_industry(ind):
        if pd.isna(ind): return "Other"
        ind = str(ind).strip()
        # 归类逻辑 (参考常见DWTS行业)
        if "actor" in ind.lower() or "actress" in ind.lower(): return "Actor"
        if "athlete" in ind.lower() or "nfl" in ind.lower() or "nba" in ind.lower() or "olympian" in ind.lower(): return "Athlete"
        if "singer" in ind.lower() or "rapper" in ind.lower() or "musician" in ind.lower(): return "Singer/Musician"
        if "model" in ind.lower(): return "Model"
        if "tv" in ind.lower() or "reality" in ind.lower() or "host" in ind.lower(): return "TV Personality"
        if "comedian" in ind.lower(): return "Comedian"
        return "Other" # 把其他小类归为 Other
        
    full_df["industry_group"] = full_df["industry"].apply(clean_industry)
    
    # 5.3 构造因变量 (同周标准化)
    
    # (A) Judge Z-Score
    # 按 season, week 分组计算 judge_score_raw 的 z-score
    print("Calculating Judge Z-Scores...")
    full_df["judge_z"] = full_df.groupby(["season", "week"])["judge_score_raw"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # (B) Fan Z-Score
    # 按 season, week 分组计算 fan_vote_mean 的 z-score
    # 注意：fan_vote_mean 是我们估算的概率，跨周不可比，必须同周比较
    print("Calculating Fan Z-Scores...")
    full_df["fan_z"] = full_df.groupby(["season", "week"])["fan_vote_mean"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # (C) Fan Logit (备选)
    # logit(p) = log(p / (1-p))
    epsilon = 1e-6
    p = full_df["fan_vote_mean"].clip(epsilon, 1-epsilon)
    full_df["fan_logit"] = np.log(p / (1 - p))
    # Logit 同样建议做一下同周中心化 (让均值为0，便于比较效应)
    full_df["fan_logit_centered"] = full_df.groupby(["season", "week"])["fan_logit"].transform(
        lambda x: x - x.mean()
    )
    
    # 6. 数据过滤
    # 确保没有关键缺失值
    initial_len = len(full_df)
    clean_df = full_df.dropna(subset=["judge_score_raw", "fan_vote_mean", "pro_dancer", "industry_group"])
    print(f"Dropped rows with missing critical data: {initial_len - len(clean_df)} rows dropped.")
    
    # 7. 保存
    clean_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved to {output_path}")
    print(f"Columns: {clean_df.columns.tolist()}")

if __name__ == "__main__":
    main()
