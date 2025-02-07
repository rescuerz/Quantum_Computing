import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.basic_provider import BasicSimulator
from qiskit import transpile

# 定义Oracle
def oracle():
    """创建Oracle电路"""
    qc = QuantumCircuit(4)
    qc.cz(0, 3)  # 在第0和第3个量子比特上施加CZ门
    return qc

# 扩散算子
def diffusion():
    """实现扩散算子"""
    qc = QuantumCircuit(4)
    
    # 对所有量子比特应用H门
    for qubit in range(4):
        qc.h(qubit)
    
    # 对所有量子比特应用X门
    for qubit in range(4):
        qc.x(qubit)
    
    # 多控制Z门（通过分解实现）
    qc.h(3)
    qc.mcx([0,1,2], 3)  # 多控制Toffoli门
    qc.h(3)
    
    # 还原X门
    for qubit in range(4):
        qc.x(qubit)
    
    # 还原H门
    for qubit in range(4):
        qc.h(qubit)
    
    return qc

# 构建完整的Grover电路
def grover_circuit(num_iterations):
    """构建完整的Grover电路"""
    qc = QuantumCircuit(4, 4)
    
    # 初始化：对所有量子比特应用H门创建均匀叠加态
    for qubit in range(4):
        qc.h(qubit)
    
    # Grover迭代
    for _ in range(num_iterations):
        # 应用Oracle
        qc.append(oracle(), range(4))
        
        # 应用扩散算子
        qc.append(diffusion(), range(4))
    
    # 测量
    qc.measure(range(4), range(4))
    
    return qc

# 计算最优迭代次数
N = 2**4  # 搜索空间大小，4个量子比特有16种状态
M = 1     # 目标态数量，假设只有一个目标态（即|0011⟩）
optimal_iterations = int(np.pi/4 * np.sqrt(N/M))
print(f"最优迭代次数: {optimal_iterations}")

# 创建和运行电路
qc = grover_circuit(optimal_iterations)
print("\n电路结构:")
print(qc.draw())

# 运行模拟
backend = BasicSimulator()
transpiled_qc = transpile(qc, backend)
result = backend.run(transpiled_qc, shots=1000).result()
counts = result.get_counts()

# 打印结果
print("\n测量结果:")
for state, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"|{state}⟩: {count} shots ({count/1000*100:.1f}%)")

# 变更目标态数量进行实验
def test_different_target_states():
    """测试不同目标态数量对Grover算法结果的影响"""
    for M in [1, 2, 4]:  # 改变目标态数量
        optimal_iterations = int(np.pi / 4 * np.sqrt(N / M))
        print(f"\n目标态数量: {M}, 最优迭代次数: {optimal_iterations}")
        
        # 创建并运行电路
        qc = grover_circuit(optimal_iterations)
        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc, shots=1000).result()
        counts = result.get_counts()
        
        # 打印结果
        print(f"测量结果 (目标态数量 {M}):")
        for state, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"|{state}⟩: {count} shots ({count/1000*100:.1f}%)")

# 测试不同目标态数量
test_different_target_states()
