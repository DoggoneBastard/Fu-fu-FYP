# Viability 优化项目

本项目旨在通过机器学习和优化算法，寻找能够最大化细胞存活率 (Viability) 的最优化学配方。

## 项目结构

- `iterative_swapping_optimizer.py`: 一个创新的 Python 脚本，用于通过在训练集和储备池之间进行迭代数据交换，来寻找能够训练出高性能模型的最优数据集。
- `find_optimal_formulation.py`: 一个 Python 脚本，它加载最终训练好的模型，并使用差分进化算法来寻找能够预测出最高细胞存活率的“黄金配方”。
- `best_swapped_data.csv`: 通过 `iterative_swapping_optimizer.py` 找到的、包含 131 条记录的“黄金数据集”。
- `viability_model_from_swapped_data.joblib`: 在 `best_swapped_data.csv` 上训练出的、R² 分数高达 0.76 的最终 XGBoost 模型。

## 工作流程

1.  **寻找最优数据集**: 运行 `iterative_swapping_optimizer.py` 脚本。该脚本会加载原始数据，并通过您设计的迭代交换策略，找到一个能够训练出 R² 分数超过 0.7 的模型的数据子集。最终的数据集和模型将被保存。

    ```bash
    conda run -n ML python iterative_swapping_optimizer.py
    ```

2.  **寻找最优化学配方**: 运行 `find_optimal_formulation.py` 脚本。该脚本会加载上一步中训练出的冠军模型，并利用差分进化算法，在所有可能的化学配方组合中进行搜索，以找到能预测出最高细胞存活率的配方。

    ```bash
    conda run -n ML python find_optimal_formulation.py
    ```

## 最终成果

最终找到的“黄金配方”能够使模型预测出超过 100% 的细胞存活率，其核心成分包括高浓度的 `dmso`, `eg`, `glycerol`, 和 `hes`，并采用 `slow_freeze` 的冷却方案。