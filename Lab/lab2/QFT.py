
from numpy import pi
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator


def qft(qc:QuantumCircuit) -> QuantumCircuit:
    for i in range(qc.num_qubits - 1, -1, -1):
        qc.h(i)
        for j in range(i - 1, -1, -1):
            qc.cp(pi / 2 ** (i - j), j, i)
    for i in range(qc.num_qubits // 2):
        qc.swap(i, qc.num_qubits - i - 1)
    return qc


# TODO: change the number of qubits
n_qubits = 2
qc = QuantumCircuit(n_qubits)
# 对qc进行操作，初态设置为10
qc.x(1)
# TODO: add quantum gate to set the initial state
qc = qft(qc)
qc.measure_all()
print(qc)


backend = BasicSimulator()
tqc = transpile(qc, backend)
result = backend.run(tqc).result()
counts = result.get_counts()
print(counts)

