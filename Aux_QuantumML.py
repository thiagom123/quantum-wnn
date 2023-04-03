import numpy as np
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.opflow import Z, I, StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient, PauliExpectation
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import  ZZFeatureMap, MCMT, RYGate, PhaseGate, RZGate
from qiskit.algorithms.optimizers import GradientDescent
from qiskit.quantum_info import partial_trace

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN, OpflowQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC



def get_opflow_qnn(quantum_circuit, feature_map, ansatz, hamiltonian, seed=42):
    algorithm_globals.random_seed = 42
    simulator = Aer.get_backend("aer_simulator", device="GPU", max_parallel_threads =0, max_parallel_experiments =0)
    quantum_instance = QuantumInstance(simulator, shots=1024, seed_simulator=algorithm_globals.random_seed, seed_transpiler=algorithm_globals.random_seed)
    qnn_expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(quantum_circuit)
    qnn = OpflowQNN(qnn_expectation, 
                    input_params=list(feature_map.parameters), 
                    weight_params=list(ansatz.parameters),
                    exp_val=PauliExpectation(),
                    gradient=Gradient(),  
                    quantum_instance=quantum_instance)
    return qnn


def apply_circuit(param_x1, param_x2, x1, x2, x3, x4):
	qc = QuantumCircuit(3)
	qc.ry(param_x1, 0)
	qc.ry(param_x2, 1)
	qc.x(0)
	qc.x(1)
	qc.append(MCMT(RYGate(x1), 2,1), [0,1,2])
	qc.x(1)
	qc.append(MCMT(RYGate(x2), 2,1), [0,1,2])
	qc.x(0)
	qc.x(1)
	qc.append(MCMT(RYGate(x3), 2,1), [0,1,2])
	qc.x(1)
	qc.append(MCMT(RYGate(x4), 2,1), [0,1,2])
	return qc

def get_statevector(x, theta, seed = 42):
    '''
    '''
    algorithm_globals.random_seed = seed
    circ = QuantumCircuit(3)
    circ.append(apply_circuit(x[0],x[1],theta[0],theta[1],theta[2],theta[3]), [0,1,2])
    backend = Aer.get_backend('statevector_simulator')
    qc_compiled = transpile(circ, backend)
    job = backend.run(qc_compiled, shots=1024, seed_simulator=algorithm_globals.random_seed, seed_transpiler=algorithm_globals.random_seed)
    result = job.result()
    state = result.get_statevector(qc_compiled, decimals = 3)
    return state


def get_bloch_coordinates(x, theta, seed=42):
    '''
    Retorna as coordenadas do vetor de Bloch, por enquanto consideramos apenas coordenadas reais.
    '''
    state_vector = get_statevector(x, theta, seed)
    density_matrix = partial_trace(state_vector, [0,1])
    a_z = 2*density_matrix.data[0][0]-1
    a_x = 2*density_matrix.data[0][1]
    return [a_x, a_z]


def ttn(dim=4):
    '''
    Implementa uma TTN Quântica
    '''
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
