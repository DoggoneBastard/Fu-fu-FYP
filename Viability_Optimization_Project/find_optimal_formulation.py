import pandas as pd
import joblib
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint

# 1. 直接从模型加载特征信息，这是最可靠的数据源
model = joblib.load('viability_model_from_swapped_data_cv.joblib')

# 从模型中获取其训练时使用的确切特征名称和顺序
feature_order = model.feature_names_in_

# 2. 根据模型的特征列表，识别不同类型的特征
chemical_features = [col for col in feature_order if col.endswith('(%)')]
cooling_rate_features = [col for col in feature_order if col.startswith('cooling rate_')]

# 从独热编码的列名中提取出原始的冷却速率类别
original_cooling_rates = sorted([feat.replace('cooling rate_', '') for feat in cooling_rate_features])

# 创建从特征名到其在模型期望列表中的索引的映射
chemical_indices = [list(feature_order).index(feat) for feat in chemical_features]
cooling_rate_indices = [list(feature_order).index(feat) for feat in cooling_rate_features]

# 3. 定义优化问题的边界
# 化学成分的边界为 0-100%
# 冷却速率的边界是一个整数选择，范围是 [0, 类别数量)
bounds = [(0, 100) for _ in chemical_features] + [(0, len(original_cooling_rates))]

# 4. 定义目标函数
def objective_function(x):
    # 创建一个长度与模型特征完全一致的零向量
    input_vector = np.zeros(len(feature_order))

    # 填充化学成分的值
    chemical_values = x[:-1]
    for i, val in enumerate(chemical_values):
        input_vector[chemical_indices[i]] = val

    # 处理冷却速率的独热编码
    # 将优化器给出的浮点数安全地转换为整数索引
    cooling_rate_choice_idx = int(np.floor(x[-1]))
    cooling_rate_choice_idx = min(cooling_rate_choice_idx, len(original_cooling_rates) - 1)
    
    # 根据选择的索引，找到对应的模型特征列的索引，并将其设为1
    target_column_index = cooling_rate_indices[cooling_rate_choice_idx]
    input_vector[target_column_index] = 1

    # 将向量转换为符合模型输入要求的DataFrame
    input_df = pd.DataFrame([input_vector], columns=feature_order)
    
    # 预测存活率 (返回负值，因为优化器是最小化)
    predicted_viability = model.predict(input_df)[0]
    return -predicted_viability

# 5. 定义约束条件：所有化学成分的总和必须在0到100之间
def sum_constraint(x):
    return np.sum(x[:-1])

nlinear_constraint = NonlinearConstraint(sum_constraint, 0, 100)

# 6. 运行差分进化优化
print("开始通过差分进化算法寻找最优配方 (已采用终极修复方案)...")
result = differential_evolution(
    objective_function,
    bounds,
    constraints=nlinear_constraint,
    strategy='best1bin',
    maxiter=200,
    popsize=30,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    disp=True,
    polish=True
)

# 7. 输出结果
print("\n--- 优化完成 ---")
# 无论是否“成功”收敛，都提取并展示搜索过程中找到的最佳结果
optimal_viability = -result.fun
optimal_formulation = result.x

print(f"找到最优解！预测最高细胞存活率 (Viability): {optimal_viability:.4f}")
print("\n--- 最优配方详情 ---")

# 化学成分
for i, feature in enumerate(chemical_features):
    print(f"{feature}: {optimal_formulation[i]:.4f}%")

# 冷却速率
optimal_cooling_rate_idx = int(np.floor(optimal_formulation[-1]))
optimal_cooling_rate_idx = min(optimal_cooling_rate_idx, len(original_cooling_rates) - 1)
optimal_cooling_rate = original_cooling_rates[optimal_cooling_rate_idx]
print(f"cooling rate: {optimal_cooling_rate}")

print(f"\n成分总和: {np.sum(optimal_formulation[:-1]):.4f}%")

if not result.success:
    print("\n--- 优化器状态 ---")
    print(f"注意: 优化算法报告未能在给定迭代次数内完全收敛。")
    print(f"原因: {result.message}")
    print("尽管如此，以上展示的是在搜索过程中发现的最佳配方。")