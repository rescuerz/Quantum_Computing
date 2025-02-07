import random #用于生成随机浮点数，以模拟量子门中的参数
import time #用于测量代码运行的时间
import matplotlib.pyplot as plt
from qubit_simulator import QubitSimulator
from qiskit.visualization import plot_histogram

def apply_circuit(circuit, n):
    # 在编号为 n-1 的量子比特上施加 Hadamard 门。
    # Hadamard 门用于将量子比特从基态 |0⟩ 或 |1⟩ 变为叠加态。
    
    circuit.h(n - 1)
    for qubit in range(n - 1):
        # 对剩下的每一对相邻的量子比特 (qubit, qubit + 1) 应用受控 U 门 (cu)
        # random.random() 生成的随机值乘以 3.14，来模拟不同的相移、旋转或任意角度操作。
        circuit.cu(qubit, qubit + 1, random.random() * 3.14, random.random() * 3.14, random.random() * 3.14)

# 记录运行时间
times = []
for i in range(2, 16):

    n_qubits = i # change this value (<=16)
    # QubitSimulator(n_qubits) 初始化量子模拟器，生成具有 n_qubits 个量子比特的系统。
    simulator = QubitSimulator(n_qubits)

    t = time.time()
    apply_circuit(simulator, n_qubits)
    run_time = time.time() - t
    print("n_qubits = {}, run_time = {}".format(n_qubits, run_time))

    times.append(run_time)

    job = simulator.run(shots=1000)
    counts = job
    print("n_qubits = {}, counts = {}".format(n_qubits, counts))
    # plot_histogram(counts).savefig("./lab1/histogram_{}.png".format(n_qubits))

print(times)
# 绘制运行时间与量子比特数的关系图
plt.figure()
plt.plot(range(2, 16), times)
plt.xlabel("n_qubits")
plt.ylabel("run_time")
plt.savefig("./lab1/run_time.png")

plt.figure()
plt.plot(range(2, 16), times, marker='o')
plt.xlabel("n_qubits")
plt.ylabel("run_time (log scale)")
plt.yscale('log')  # 设置 y 轴为对数刻度
plt.grid(True, which="both", ls="--")  # 添加网格线便于阅读对数图
plt.savefig("./lab1/run_time_log_scale.png")
