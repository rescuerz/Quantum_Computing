import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, QFT
from qiskit.providers.basic_provider import BasicSimulator


def shor_circuit(N, a, n_p, n_v):
    qc = QuantumCircuit(n_p + n_v, n_p)
    # 初始化period register
    # 对每一个量子比特执行Hadamard变换
    for q in range(n_p):
        qc.h(q)
    qc.x(n_p)
    # 对每一个量子比特执行控制模指数电路
    for q in range(n_p):
        exponent = 2 ** q
        ctrl_mod = mod_exp_circuit(a, exponent, N, n_v).to_gate().control(1)
        qc.append(ctrl_mod, [q] + list(range(n_p, n_p + n_v)))
    
    qc.append(QFT(n_p, inverse=True), range(n_p))

    qc.measure(range(n_p), range(n_p))
    return qc

def mod_exp_circuit(a, power, N, n_v):
    qc = QuantumCircuit(n_v)
    # 施加power次模指数电路
    for _ in range(power):
        qc.append(mod_circuit(a, N, n_v), range(n_v))
    return qc

# def mod_circuit(a, N, n_v):
#     matrix = np.zeros((2 ** n_v, 2 ** n_v), dtype=int)
#     # TODO complete modular multiplication circuit
    
def mod_circuit(a, N, n_v):
    """
    Create a quantum circuit for modular multiplication: |x⟩ -> |ax mod N⟩
    """
    matrix = np.zeros((2 ** n_v, 2 ** n_v), dtype=complex)
    
    # Fill the matrix with the modular multiplication results
    for x in range(2 ** n_v):
        # 如果x小于N，则计算(a*x) mod N
        if x < N:
            y = (a * x) % N
            matrix[y, x] = 1
            # matrix[x, y] = 1
        else:
            matrix[x, x] = 1
            
    # Create and return a unitary gate from the matrix
    return UnitaryGate(matrix)
# 对于每个小于N的输入x，计算 f(x) = (2x) mod 5：
# x = 0: f(0) = (2 × 0) mod 5 = 0
# x = 1: f(1) = (2 × 1) mod 5 = 2
# x = 2: f(2) = (2 × 2) mod 5 = 4
# x = 3: f(3) = (2 × 3) mod 5 = 1
# x = 4: f(4) = (2 × 4) mod 5 = 3
# x ≥ 5: f(x) = x （保持不变）
# |0> <0| + |1> <2| + |2> <4| + |3> <1| + |4> <3| + |5> <5| + |6> <6| + |7> <7|
#     |0⟩ |1⟩  |2⟩  |3⟩ |4⟩ |5⟩  |6⟩ |7⟩
# |0⟩  1   0   0   0   0   0   0   0
# |1⟩  0   0   0   1   0   0   0   0
# |2⟩  0   1   0   0   0   0   0   0
# |3⟩  0   0   0   0   1   0   0   0
# |4⟩  0   0   1   0   0   0   0   0
# |5⟩  0   0   0   0   0   1   0   0
# |6⟩  0   0   0   0   0   0   1   0
# |7⟩  0   0   0   0   0   0   0   1

N = 21
# 随机取小于n且与n互质的正整数a
a = 8
# 
n_p = 3 # number of qubits in period register
n_v = 5 # number of qubits in value register


print(f"a: {a}, n_p: {n_p}")
qc = shor_circuit(N, a, n_p, n_v)
print(qc.draw())

backend = BasicSimulator()
tqc = transpile(qc, backend)
result = backend.run(tqc).result()
counts = result.get_counts()
print("counts:", counts)

r = len(counts)
print(f"r: {r}")

# 如果r是偶数, 则令x = a^(r/2)， 有 x^2 = 1 mod N, (x-1)(x+1) = 0 mod N
# 从而x-1和x+1是N的因子
# 要求x+1 不等于 N，才能进行下一步操作
if r % 2 == 0 and pow(a, r // 2, N) != N - 1:
    factor1 = np.gcd(pow(a, r // 2) - 1, N)
    factor2 = np.gcd(pow(a, r // 2) + 1, N)
    print(f"{N} = {factor1} * {factor2}")
else:
    print("Invalid a!")