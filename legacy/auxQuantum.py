import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.opflow import Z, I, StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient, PauliExpectation
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import  ZZFeatureMap, MCMT, RYGate, PhaseGate, RZGate
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library.standard_gates import HGate

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN, OpflowQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR

from typing import Union

from qiskit_machine_learning.exceptions import QiskitMachineLearningError


algorithm_globals.random_seed = 42
hamiltonian = Z ^ I^ I ^ I^ I^ I^ I


def FeatureMap():
	dim = 2
	param_x1 = Parameter("x1")
	param_x2 = Parameter("x2")
	#param_x3 = Parameter("x3")
	#param_x4 = Parameter("x4")

	feature_map3 = QuantumCircuit(dim)
	feature_map3.ry(param_x1, 0)
	feature_map3.ry(param_x2, 1)
	#feature_map3.ry(param_x3, 2)
	#feature_map3.ry(param_x4, 3)
	return feature_map3


def AngleEncoding4Qubits():
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


def TTN():
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


def Neuron(name1, name2, name3, name4):
	x1 = Parameter(name1)
	x2 = Parameter(name2)
	x3 = Parameter(name3)
	x4 = Parameter(name4)
	qr = QuantumRegister(3)
	qc = QuantumCircuit(qr)
	qc.append(MCMT(RYGate(x1), 2,1), [0,1,2])
	qc.x(0)
	qc.append(MCMT(RYGate(x2), 2,1), [0,1,2])
	qc.x(1)
	qc.append(MCMT(RYGate(x3), 2,1), [0,1,2])
	qc.x(0)
	qc.append(MCMT(RYGate(x4), 2,1), [0,1,2])
	qc.x(1)
	return qc


def NeuronModified(name1, name2, name3, name4):
	x1 = Parameter(name1)
	x2 = Parameter(name2)
	x3 = Parameter(name3)
	x4 = Parameter(name4)
	qr = QuantumRegister(3)
	qc = QuantumCircuit(qr)
	qc.h(2)
	qc.append(MCMT(RYGate(x1), 2,1), [0,1,2])
	qc.x(0)
	qc.append(MCMT(RYGate(x2), 2,1), [0,1,2])
	qc.x(1)
	qc.append(MCMT(RYGate(x3), 2,1), [0,1,2])
	qc.x(0)
	qc.append(MCMT(RYGate(x4), 2,1), [0,1,2])
	qc.x(1)
	return qc


def WeightlessNN():
	qr = QuantumRegister(7)
	qc = QuantumCircuit(qr)
	qc.append(Neuron("p1", "p2", "p3", "p4"), [0,1,2])
	qc.append(Neuron("q1", "q2", "q3", "q4"), [3,4,5])
	qc.append(Neuron("s1", "s2", "s3", "s4"), [2,5,6])

	return qc

def WeightlessNNMod():
	qr = QuantumRegister(7)
	qc = QuantumCircuit(qr)
	qc.append(NeuronModified("p1", "p2", "p3", "p4"), [0,1,2])
	qc.append(NeuronModified("q1", "q2", "q3", "q4"), [3,4,5])
	qc.append(NeuronModified("s1", "s2", "s3", "s4"), [2,5,6])

	return qc

