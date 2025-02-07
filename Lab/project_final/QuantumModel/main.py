from tests.test import run_classification_test, run_quantum_test
import pandas as pd

if __name__ == '__main__':
    # 定义要测试的模型组合
    # combinations = [
    #     ('C', 'Q'), ('Q', 'Q')
    # ]
    # combinations = [
    #     ('C', 'Q')
    # ]
    combinations = [
        ('C', 'C'), ('C', 'Q')
    ]
    
    # 存储所有结果
    all_results = []
    
    # 测试不同的分类数量
    for num_classes in [2, 4, 10]:
        results = run_classification_test(num_classes, combinations)
        all_results.extend(results)
        results = run_quantum_test(num_classes, combinations)
        all_results.extend(results)
    
    # 创建结果DataFrame并显示
    results_df = pd.DataFrame(all_results)
    print("\n所有测试结果汇总:")
    print(results_df.to_string(index=False))
    
    # 保存结果到CSV文件
    results_df.to_csv('results1.csv', index=False)
    print("\n结果已保存到 results1.csv")