from qubit_simulator import QubitSimulator
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# use Aer's AerSimulator
simulator = AerSimulator()
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(5, 5)
# Add a H gate on qubit 0
circuit.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.cx(2, 3)
circuit.cx(3, 4)

circuit.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
# Draw the circuit
circuit.draw("mpl").savefig("./lab1/circuit.png")
#  compile the circuit for the simulator
compiled_circuit = transpile(circuit, simulator)
# execute the circuit on the simulator
job = simulator.run(compiled_circuit, shots=1000)
# get the result from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
# 绘制直方图
plot_histogram(counts)
# 设置 x 轴标签水平显示
plt.xticks(rotation=0)
# 保存直方图
plt.savefig("./lab1/histogram.png")
print("\n Total count for 00000 and 11111 are:", counts)

# 用于生成 5 个量子比特的量子模拟器
n_qubits = 5
simulator = QubitSimulator(n_qubits)
simulator.h(0)
simulator.cx(0, 1)
simulator.cx(1, 2)
simulator.cx(2, 3)
simulator.cx(3, 4)
counts = simulator.run(shots=10000)
print(simulator)
print(counts)