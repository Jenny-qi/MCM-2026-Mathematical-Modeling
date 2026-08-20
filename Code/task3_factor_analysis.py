import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings

# 忽略收敛警告 (数据量较小或结构复杂时可能出现)
warnings.filterwarnings("ignore")

def fit_lmm(df, formula, model_name):
    """
    拟合 LMM 并返回模型对象与结果摘要
    """
    print(f"\n--- Fitting {model_name} ---")
    print(f"Formula: {formula}")
    
    # 使用 statsmodels 的 mixedlm
    # 注意：statsmodels 的 MixedLM 公式接口对交叉随机效应支持不如 lme4 直观
    # 这里我们使用 variance components 结构：
    # groups=season (作为主分组，其实在这里我们主要想要 pro_dancer 和 contestant 的随机效应)
    # 更好的做法是把整个数据集视为一个 group，然后加两个 VC (Variance Component)
    
    # 由于 statsmodels 对交叉随机效应 (Crossed Random Effects) 支持有限，
    # 我们采用 "re_formula" + "vc_formula" 的方式，或者把所有数据设为同一组
    
    df["group"] = 1 # 伪分组，用于容纳交叉随机效应
    
    # 定义方差分量：Pro Dancer 和 Contestant
    # vc_formula = {"pro": "0 + C(pro_dancer)", "star": "0 + C(contestant)"}
    # 上面的写法在类别很多时会非常慢且内存爆炸。
    
    # 替代方案：
    # 考虑到 Python statsmodels 的性能限制，我们这里做简化处理：
    # 随机效应主要关注 pro_dancer。contestant 的个体差异我们尝试放入 VC。
    # 如果跑不动，我们退而求其次：
    # 将 contestant 作为 Nested Random Effect (如果 pro 不换人)，但 pro 换人。
    
    # 实用策略：
    # 主随机效应：pro_dancer (我们最关心的)
    # 那个 "contestant" 个体差异，我们尝试用 Fixed Effect (如果不太多) 或者
    # 忽略 (假设 age/industry 已解释大部分)，或者
    # 使用 vc_formula={"star": "0 + C(contestant)"}
    
    # 让我们尝试包含两个 VC。如果内存不够，代码会捕获异常。
    # 为了速度，我们先只放 pro_dancer 的随机效应，看看能不能跑通。
    # 毕竟题目核心问的是 "Impact of Pro Dancers".
    
    try:
        # 模型 A: 只有 Pro 的随机效应
        # model = smf.mixedlm(formula, df, groups=df["pro_dancer"]) 
        # 不对，pro_dancer 是 grouping factor。
        # 这样只能得到 pro 之间的方差，不能得到每个 pro 的 BLUP (虽然也能算，但结构不对)
        # 正确做法：groups=pro_dancer, re_formula="~1" -> 随机截距
        
        # 但我们还需要 contestant 随机效应。
        # 让我们用 vc_formula。
        
        vc = {"pro": "0 + C(pro_dancer)", "star": "0 + C(contestant)"}
        model = smf.mixedlm(formula, df, groups="group", vc_formula=vc)
        result = model.fit(reml=True, method='lbfgs') # 使用 LBFGS 优化
        
        return result
        
    except Exception as e:
        print(f"Model fitting failed: {e}")
        return None

def extract_fixed_effects(result, model_label):
    """提取固定效应系数"""
    if result is None: return pd.DataFrame()
    
    fe = result.params.to_frame(name="coef")
    fe["std_err"] = result.bse
    fe["z_value"] = result.tvalues
    fe["p_value"] = result.pvalues
    fe["ci_lower"] = result.conf_int()[0]
    fe["ci_upper"] = result.conf_int()[1]
    fe["model"] = model_label
    
    # 过滤掉方差参数 (Var ...) 和 随机效应均值 (Intercept RE)
    # statsmodels 结果里包含 Group Var 等
    fe = fe[~fe.index.str.contains("Var") & ~fe.index.str.contains("Group")]
    
    # 重新整理索引名
    fe.index.name = "term"
    return fe.reset_index()

def extract_random_effects_blup(result, vc_name="pro"):
    """
    提取指定方差分量 (Variance Component) 的随机效应估计 (BLUP)
    statsmodels 的 random_effects 属性返回的是 {group_id: {term: value}}
    对于 vc_formula，它会返回每个 group (这里是 '1') 下的随机效应向量。
    """
    if result is None: return pd.DataFrame()
    
    # result.random_effects 是一个字典，键是 group label (这里全是 1)
    re_dict = result.random_effects[1] # 取出 group=1 的结果
    
    # re_dict 的键大概长这样: "pro[T.Cheryl Burke]"
    # 我们需要解析它
    
    records = []
    for term, value in re_dict.items():
        if term.startswith(vc_name):
            # 解析名字 pro[T.Name] -> Name
            # 通常格式是 "pro[T.Valentin Chmerkovskiy]"
            import re
            match = re.search(f"{vc_name}\[T\.(.*?)\]", term)
            if match:
                name = match.group(1)
                records.append({"name": name, "blup": value})
            else:
                # 可能是 "pro" (如果没用 Categorical)
                records.append({"name": term, "blup": value})
                
    return pd.DataFrame(records)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "task3_features.csv")
    
    # 输出路径
    output_coef_path = os.path.join(base_dir, "task3_fixed_effects.csv")
    output_pro_path = os.path.join(base_dir, "task3_pro_effects.csv")
    
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 定义公式
    # 固定效应：Age (标准化), Industry (分类)
    # 我们去掉 Intercept (或者保留)，这里保留默认 Intercept
    # C(industry_group, Treatment(reference='Actor')) 设 Actor 为基准
    fixed_formula = " ~ age_std + C(industry_group, Treatment(reference='Actor'))"
    
    # 1. 拟合 Judge 模型 (因变量 judge_z)
    judge_formula = "judge_z" + fixed_formula
    judge_res = fit_lmm(df, judge_formula, "Judge_Model")
    
    # 2. 拟合 Fan 模型 (因变量 fan_z)
    fan_formula = "fan_z" + fixed_formula
    fan_res = fit_lmm(df, fan_formula, "Fan_Model")
    
    # 3. 提取结果
    
    # (A) 固定效应
    fe_judge = extract_fixed_effects(judge_res, "Judge")
    fe_fan = extract_fixed_effects(fan_res, "Fan")
    
    all_fe = pd.concat([fe_judge, fe_fan], ignore_index=True)
    all_fe.to_csv(output_coef_path, index=False)
    print(f"\nFixed effects saved to {output_coef_path}")
    print(all_fe[["term", "model", "coef", "p_value"]])
    
    # (B) Pro 随机效应 (BLUP)
    re_pro_judge = extract_random_effects_blup(judge_res, "pro")
    re_pro_judge = re_pro_judge.rename(columns={"blup": "judge_effect"})
    
    re_pro_fan = extract_random_effects_blup(fan_res, "pro")
    re_pro_fan = re_pro_fan.rename(columns={"blup": "fan_effect"})
    
    # 合并 Pro 效应
    if not re_pro_judge.empty and not re_pro_fan.empty:
        pro_effects = pd.merge(re_pro_judge, re_pro_fan, on="name", how="outer")
        
        # 补充统计信息：每个 Pro 带过的周数 (Sample Size)
        pro_counts = df["pro_dancer"].value_counts().reset_index()
        pro_counts.columns = ["name", "n_weeks"]
        
        # 补充：带过的明星数
        pro_stars = df.groupby("pro_dancer")["contestant"].nunique().reset_index()
        pro_stars.columns = ["name", "n_stars"]
        
        pro_effects = pd.merge(pro_effects, pro_counts, on="name", how="left")
        pro_effects = pd.merge(pro_effects, pro_stars, on="name", how="left")
        
        # 排序
        pro_effects = pro_effects.sort_values("judge_effect", ascending=False)
        
        pro_effects.to_csv(output_pro_path, index=False)
        print(f"Pro effects saved to {output_pro_path}")
        print(pro_effects.head())
    else:
        print("Warning: Failed to extract random effects.")

if __name__ == "__main__":
    main()
