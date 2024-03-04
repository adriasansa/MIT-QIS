#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:08:52 2024

@author: adria
"""

import pennylane as qml
from pennylane import numpy as np

wires = 2

dev = qml.device('default.qubit',wires=wires)
@qml.qnode(dev)

def circuit_qft(basis_state):
    qml.BasisState(basis_state, wires=range(wires))
    qml.QFT(wires=range(wires))
    #Now inverse QFT
    qml.SWAP((0,1))
    qml.Hadamard(1)
    qml.ctrl(qml.adjoint(qml.S(0)), (1))
    qml.Hadamard(0)
    return qml.state()

Psi_0 = np.array([0,1], requires_grad=False)

QFT = circuit_qft(Psi_0)
print("QFT")
print(QFT)

dev_ = qml.device('default.qubit',wires=wires)
@qml.qnode(dev_)

def circuit_qft_manual(basis_state):
    qml.BasisState(basis_state, wires=range(wires))
    qml.Hadamard(0)
    qml.ctrl(qml.S(0), (1))
    qml.Hadamard(1)
    qml.SWAP((0,1))
    return qml.state()

QFT_manual = circuit_qft_manual(Psi_0)

print("QFT manual")
print(qml.draw(circuit_qft_manual)(Psi_0))
print(QFT_manual)