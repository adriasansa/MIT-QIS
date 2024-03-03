# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:08:14 2024

@author: Adrià Sansa Perna
"""
import pennylane as qml
from pennylane import numpy as np

wires = 3

dev = qml.device('default.qubit',wires=wires)
@qml.qnode(dev)

def circuit_qft(basis_state):
    qml.BasisState(basis_state, wires=range(wires))
    qml.Hadamard(wires=[0])
    qml.Hadamard(wires=[1])
    qml.Hadamard(wires=[2])
    qml.QFT(wires=range(wires))
    return qml.state()

QFT = circuit_qft(np.array([0,0,0], requires_grad=False))

print("QFT")
print(QFT)

### Result of openMIT SU3 QFT 1st problem
