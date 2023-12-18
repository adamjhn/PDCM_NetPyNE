from neuron import h
from cfgSS import cfg
import numpy as np


class L2e:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L2i:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L4e:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L4i:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L5e:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L5i:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L6e:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


class L6i:
    def __init__(self):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        self.soma.cm = cfg.Cm
        self.soma.Ra = cfg.Ra
        if cfg.epas:
            self.soma.insert("pas")
            self.soma(0.5).pas.e = cfg.epas
            self.soma(0.5).pas.g = cfg.gpas


# v0.00 - classes for each cell type in network
