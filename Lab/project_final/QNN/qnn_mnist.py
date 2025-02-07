import importlib, pkg_resources
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# 常量定义
THRESHOLD = 0.5
EPOCHS = 3
BATCH_SIZE = 32

class DataPreprocessor:
    @staticmethod
    def load_and_preprocess_data():
        """加载并预处理MNIST数据集"""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
        return x_train, y_train, x_test, y_test

    @staticmethod
    def filter_36(x, y):
        """过滤出数字3和6的样本"""
        keep = (y == 3) | (y == 6)
        x, y = x[keep], y[keep]
        y = y == 3
        return x, y

    @staticmethod
    def remove_contradicting(xs, ys):
        """删除矛盾的样本"""
        mapping = collections.defaultdict(set)
        orig_x = {}
        for x, y in zip(xs, ys):
            orig_x[tuple(x.flatten())] = x
            mapping[tuple(x.flatten())].add(y)
        
        new_x, new_y = [], []
        for flatten_x in mapping:
            x = orig_x[flatten_x]
            labels = mapping[flatten_x]
            if len(labels) == 1:
                new_x.append(x)
                new_y.append(next(iter(labels)))
        
        return np.array(new_x), np.array(new_y)

class QuantumCircuitBuilder:
    @staticmethod
    def convert_to_circuit(image):
        """将图像转换为量子电路"""
        values = np.ndarray.flatten(image)
        qubits = cirq.GridQubit.rect(4, 4)
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            if value:
                circuit.append(cirq.X(qubits[i]))
        return circuit

    class CircuitLayerBuilder:
        def __init__(self, data_qubits, readout):
            self.data_qubits = data_qubits
            self.readout = readout
        
        def add_layer(self, circuit, gate, prefix):
            for i, qubit in enumerate(self.data_qubits):
                symbol = sympy.Symbol(prefix + '-' + str(i))
                circuit.append(gate(qubit, self.readout)**symbol)

class ModelBuilder:
    @staticmethod
    def create_quantum_model():
        """创建量子神经网络模型"""
        data_qubits = cirq.GridQubit.rect(4, 4)
        readout = cirq.GridQubit(-1, -1)
        circuit = cirq.Circuit()
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))
        
        builder = QuantumCircuitBuilder.CircuitLayerBuilder(data_qubits, readout)
        builder.add_layer(circuit, cirq.XX, "xx1")
        builder.add_layer(circuit, cirq.ZZ, "zz1")
        circuit.append(cirq.H(readout))
        
        return circuit, cirq.Z(readout)

    @staticmethod
    def create_classical_model():
        """创建经典CNN模型"""
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1),
        ])

    @staticmethod
    def create_fair_classical_model():
        """创建公平的经典模型"""
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

class Trainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.quantum_builder = QuantumCircuitBuilder()
        self.model_builder = ModelBuilder()

    def train_all_models(self):
        # 加载和预处理数据
        x_train, y_train, x_test, y_test = self.preprocessor.load_and_preprocess_data()
        x_train, y_train = self.preprocessor.filter_36(x_train, y_train)
        x_test, y_test = self.preprocessor.filter_36(x_test, y_test)

        # 准备量子模型数据
        x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
        x_test_small = tf.image.resize(x_test, (4, 4)).numpy()
        x_train_nocon, y_train_nocon = self.preprocessor.remove_contradicting(x_train_small, y_train)
        
        x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
        x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

        # 转换为量子电路
        x_train_circ = [self.quantum_builder.convert_to_circuit(x) for x in x_train_bin]
        x_test_circ = [self.quantum_builder.convert_to_circuit(x) for x in x_test_bin]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

        # 训练量子模型
        circuit, readout = self.model_builder.create_quantum_model()
        qnn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, readout),
        ])
        
        y_train_hinge = 2.0 * y_train_nocon - 1.0
        y_test_hinge = 2.0 * y_test - 1.0
        
        qnn_model.compile(
            loss=tf.keras.losses.Hinge(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])
        
        qnn_history = qnn_model.fit(
            x_train_tfcirc, y_train_hinge,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_test_tfcirc, y_test_hinge))
        
        qnn_results = qnn_model.evaluate(x_test_tfcirc, y_test)

        # 训练经典CNN模型
        cnn_model = self.model_builder.create_classical_model()
        cnn_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])
        
        cnn_history = cnn_model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=20,
            validation_data=(x_test, y_test))
        
        cnn_results = cnn_model.evaluate(x_test, y_test)

        # 训练公平经典模型
        fair_model = self.model_builder.create_fair_classical_model()
        fair_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])
        
        fair_history = fair_model.fit(
            x_train_bin, y_train_nocon,
            batch_size=128,
            epochs=20,
            validation_data=(x_test_bin, y_test))
        
        fair_results = fair_model.evaluate(x_test_bin, y_test)

        # 修改返回值，加入训练历史
        return {
            'accuracy': {
                'qnn': qnn_results[1],
                'cnn': cnn_results[1],
                'fair': fair_results[1]
            },
            'history': {
                'qnn': qnn_history.history,
                'cnn': cnn_history.history,
                'fair': fair_history.history
            }
        }

def plot_results(results):
    """绘制比较结果"""
    # 绘制准确率柱状图
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=["Quantum", "Classical, full", "Classical, fair"],
        y=[results['accuracy']['qnn'], 
           results['accuracy']['cnn'], 
           results['accuracy']['fair']])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_comparison.png')
    plt.close()

    # 为每个模型分别绘制训练历史
    model_names = ['qnn', 'cnn', 'fair']
    display_names = {'qnn': 'Quantum Neural Network', 
                    'cnn': 'Classical CNN', 
                    'fair': 'Fair Classical Model'}

    for model in model_names:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 准确率曲线
        ax1.plot(results['history'][model]['accuracy'], 'b-', label='Accuracy')
        ax1.set_title(f'{display_names[model]} - Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # 损失曲线
        ax2.plot(results['history'][model]['loss'], 'r-', label='Loss')
        ax2.set_title(f'{display_names[model]} - Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'training_history_{model}.png')
        plt.close()

def main():
    trainer = Trainer()
    results = trainer.train_all_models()
    plot_results(results)

if __name__ == "__main__":
    main()