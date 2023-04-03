from qiskit import QuantumCircuit
from qiskit.circuit import Parameter



def angle_encoding(input_size, parameter_label = "E"):
	'''
	Applies Angle Encoding in N qubits
	input_size: Size of the input
	parameter_label: label for the parameters, it may be required to use more than
	different labels if the function is applied more than once in the circuit.
	
	return: Quantum Circuit with Angle Encoding in 2 qubits
	'''
	dim = input_size

	x = []
	for k in range(input_size):
		x.append(Parameter(parameter_label + str(k)))
	qc = QuantumCircuit(input_size)

	for k in range(input_size):
		qc.ry(x[k], k)

	return qc


'''
def angle_encoding_2qubits():
	dim = 2
	param_x1 = Parameter("x1")
	param_x2 = Parameter("x2")

	feature_map3 = QuantumCircuit(dim)
	feature_map3.ry(param_x1, 0)
	feature_map3.ry(param_x2, 1)
	return feature_map3


def angle_encoding_4qubits():
	dim = 4
	param_x1 = Parameter("x1")
	param_x2 = Parameter("x2")
	param_x3 = Parameter("x3")
	param_x4 = Parameter("x4")

	feature_map3 = QuantumCircuit(dim)
	feature_map3.ry(param_x1, 0)
	feature_map3.ry(param_x2, 1)
	feature_map3.ry(param_x3, 2)
	feature_map3.ry(param_x4, 3)
	return feature_map3
'''
'''
def ttn():
	dim = 4
	ansatz = QuantumCircuit(dim)
	param_y1 = Parameter("y1")
	param_y2 = Parameter("y2")
	param_y3 = Parameter("y3")
	param_y4 = Parameter("y4")
	ansatz.ry(param_y1, 0)
	ansatz.ry(param_y2, 1)
	ansatz.ry(param_y3, 2)
	ansatz.ry(param_y4, 3)
	ansatz.cx(0,1)
	ansatz.cx(3,2)
	param_y21 = Parameter("y21")
	param_y31 = Parameter("y31")
	ansatz.ry(param_y21, 1)
	ansatz.ry(param_y31, 2)
	ansatz.cx(1,2)
	return ansatz
'''
