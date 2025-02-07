from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# use Aer's AerSimulator
simulator = AerSimulator()
# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)
# Add a H gate on qubit 0
circuit.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
# map the quantum measurement to the classical bits
circuit.measure([0, 1], [0, 1])
# Draw the circuit
circuit.draw("mpl").savefig("./lab0/circuit.png")
#  compile the circuit for the simulator
compiled_circuit = transpile(circuit, simulator)
# execute the circuit on the simulator
job = simulator.run(compiled_circuit, shots=1000)
# get the result from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
plot_histogram(counts).savefig("./lab0/histogram.png")
print("\n Total count for 00 and 11 are:", counts)