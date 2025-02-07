from numpy import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import QFT

def qft(qc: QuantumCircuit) -> QuantumCircuit:
    for i in range(qc.num_qubits - 1, -1, -1):
        qc.h(i)
        for j in range(i - 1, -1, -1):
            qc.cp(pi / 2 ** (i - j), j, i)
    for i in range(qc.num_qubits // 2):
        qc.swap(i, qc.num_qubits - i - 1)
    return qc

def create_qpe_circuit(n_count: int, theta: float) -> QuantumCircuit:
    # 创建量子和经典寄存器
    qr = QuantumRegister(n_count + 1)  # n_count个计数比特 + 1个目标比特
    cr = ClassicalRegister(n_count)
    qc = QuantumCircuit(qr, cr)
    
    # 初始化目标比特为|1⟩
    qc.x(n_count)
    
    # 对计数寄存器应用H门
    for i in range(n_count):
        qc.h(i)
    
    # 应用受控P门
    for i in range(n_count):
        for j in range(2**i):  # 每个比特位需要重复2^i次
            qc.cp(theta, i, n_count)
    
    # 应用逆QFT到计数寄存器
    qc = qc.compose(QFT(n_count, inverse=True), list(range(n_count)))
    
    # 测量计数寄存器
    qc.measure(list(range(n_count)), list(range(n_count)))
    
    return qc

# 测试θ=2π/3的情况
print("\nTesting with θ=2π/3:")
n_count = 3  # 使用3个计数比特
theta = 2*pi/3
qc = create_qpe_circuit(n_count, theta)
print("Circuit:")
print(qc)

backend = BasicSimulator()
tqc = transpile(qc, backend)
result = backend.run(tqc, shots=1000).result()
counts = result.get_counts()
print("\nMeasurement results (θ=2π/3):")
print(counts)

