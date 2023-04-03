from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def quantum_weightless_neuron_ry(input_size, parameter_label='p'):
	"""
	input_size: Number of input qubits
	parameter_label: label for the parameters, it may be required to use more than
	different labels if the function is applied more than once in the circuit.

	return: Quantum Circuit with Quantum Weightless Neural Network (Ry gate) 
	with input_size+1 qubits (1 qubit of ouput)
	"""
	x = []
	for k in range(2 ** input_size):
		x.append(Parameter(parameter_label + str(k)))
	qc = QuantumCircuit(input_size+1)
	ctrl_qubits=[]
	for i in range(input_size):
		ctrl_qubits.append(i)

	for k in range(2 ** input_size):

		for l in range(input_size):
			c = 2 ** l
			if(k & c == c):
				qc.x(l)
		
		qc.mcry(theta  = x[k], q_controls =ctrl_qubits, q_target = input_size)
		for l in range(input_size):
			c = 2 ** l
			if(k & c == c):
				qc.x(l)
	return qc


def quantum_weightless_nn():
	"""
	Creates a 2 layers network for 4 input qubits
	"""
	qc = QuantumCircuit(7)
	qc.append(quantum_weightless_neuron_ry(2, "p"), [0,1,2])
	qc.append(quantum_weightless_neuron_ry(2, "s"), [3,4,5])
	qc.append(quantum_weightless_neuron_ry(2, "t"), [2,5,6])

	return qc



