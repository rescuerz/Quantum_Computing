import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from models.param_gen import GPOParamGen
from models.param_gen import optimize_parameters
from models.classical_model import ClassicalModel
from models.hybrid_model import HybridModel
from data.mnist_dataset import MNISTDataset


def run_classification_test(num_classes, combinations):
    print(f"\n{'='*20} 测试{num_classes}分类 {'='*20}")
    
    # 创建对应类别数的数据集
    mnist_dataset = MNISTDataset(num_classes=num_classes)
    print(f"\n数据集信息:")
    train_data, train_labels = mnist_dataset.getTrainData()
    test_data, test_labels = mnist_dataset.getTestData()
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")

    # 超参数优化
    bounds_params = {
        "nb_hidden_neurons": [1, 30, 1]
    }

    gpo = GPOParamGen(bounds_params, max_itr=30)
    hp = optimize_parameters(
        lambda **kwargs: ClassicalModel(output_shape=num_classes, **kwargs),
        *mnist_dataset.getTrainData(), 
        gpo, 
        fit_kwargs={"epochs": 20}
    )
    print(f"\n优化后的超参数: {hp}")
    # gpo.show_expectation()

    # 测试经典模型
    results = []
    
    c_model = ClassicalModel(output_shape=num_classes, **hp)
    print("\n测试经典模型:")
    print(c_model)
    history_c = c_model.fit(
        *mnist_dataset.getTrainData(),
        *mnist_dataset.getValidationData(),
        batch_size=32,
        verbose=True
    )
    test_score = c_model.score(*mnist_dataset.getTestData())
    print(f"经典模型测试分数: {test_score:.4f}")
    # 在result.csv文件中添加一行
    with open('result.csv', 'a') as f:
        f.write(f"classical_{num_classes}_class\n")
    c_model.show_history(history_c, name=f'classical_{num_classes}_class')
    # 在result.csv文件中添加一行
    with open('result.csv', 'a') as f:
        f.write(f"Classical test score: {test_score:.4f}\n")
    # 收集结果
    results.append({
        'model': 'Classical',
        'num_classes': num_classes,
        'test_score': test_score
    })
    return results

def run_quantum_test(num_classes, combinations):
    print(f"\n{'='*20} 测试{num_classes}分类 {'='*20}")
    # 创建对应类别数的数据集
    mnist_dataset = MNISTDataset(num_classes=num_classes)
    print(f"\n数据集信息:")
    train_data, train_labels = mnist_dataset.getTrainData()
    test_data, test_labels = mnist_dataset.getTestData()
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")

    # 超参数优化
    bounds_params = {
        "nb_hidden_neurons": [1, 30, 1]
    }

    gpo = GPOParamGen(bounds_params, max_itr=30)
    hp = optimize_parameters(
        lambda **kwargs: ClassicalModel(output_shape=num_classes, **kwargs),
        *mnist_dataset.getTrainData(), 
        gpo, 
        fit_kwargs={"epochs": 20}
    )
    print(f"\n优化后的超参数: {hp}")
    # gpo.show_expectation()

    # 测试混合模型
    results = []
    print("\n测试混合模型:")
    for backbone_type, classifier_type in combinations:
        model_name = f"Hybrid_{backbone_type}{classifier_type}"
        print(f"\n测试 {model_name}")
        
        hybrid_model = HybridModel(
            input_shape=(1, 8, 8),
            output_shape=num_classes,
            backbone_type=backbone_type,
            classifier_type=classifier_type,
            **hp
        )
        
        history_h = hybrid_model.fit(
            *mnist_dataset.getTrainData(),
            *mnist_dataset.getValidationData(),
            batch_size=32,
            verbose=True
        )
        
        test_score = hybrid_model.score(*mnist_dataset.getTestData())
        print(f"{model_name} 测试分数: {test_score:.4f}")
        
        # 保存训练历史
        # 在result.csv文件中添加一行
        with open('result.csv', 'a') as f:
            f.write(f"{model_name}_{num_classes}_class\n")
        hybrid_model.show_history(history_h, name=f'{model_name}_{num_classes}_class')
        # 在result.csv文件中添加一行
        with open('result.csv', 'a') as f:
            f.write(f"{model_name} test score: {test_score:.4f}\n")
        # 收集结果
        results.append({
            'model': model_name,
            'num_classes': num_classes,
            'test_score': test_score
        })

    return results

if __name__ == '__main__':
    # 定义要测试的模型组合
    combinations = [
        ('C', 'Q'), ('Q', 'Q')
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
    results_df.to_csv('classification_results.csv', index=False)
    print("\n结果已保存到 classification_results.csv")