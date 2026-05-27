from neuron import h
import numpy as np

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from cfgSS import (
        cfg,
    )  # if no simConfig in parent module, import directly from cfg.py:cfg


class SimpleNeuron:
    def __init__(self, dendL=None, dendR=None):
        self.soma = h.Section(name="soma", cell=self)
        # add 3D points to locate the neuron in the ECS
        self.soma.pt3dadd(0.0, 0.0, 0.0, 2.0 * cfg.somaR)
        self.soma.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, 2.0 * cfg.somaR)
        if dendL is not None:
            self.dend = h.Section(name="dend", cell=self)
            self.dend.connect(self.soma(1), 0)
            self.dend.pt3dadd(0.0, 2.0 * cfg.somaR, 0.0, dendR)
            self.dend.pt3dadd(0.0, 2.0 * cfg.somaR + dendL, 0.0, dendR)
            allsec = [self.soma, self.dend]
        else:
            allsec = [self.soma]
        for sec in allsec:
            sec.cm = cfg.Cm
            sec.Ra = cfg.Ra
            if cfg.epas:
                sec.insert("pas")
                for seg in sec:
                    seg.pas.e = cfg.epas
                    seg.pas.g = cfg.gpas


class L2e(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L2i(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L4e(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L4i(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L5e(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L5i(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L6e(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


class L6i(SimpleNeuron):
    def __init__(self):
        dendL = getattr(cfg, "dendL", None)
        dendR = getattr(cfg, "dendR", None)
        super().__init__(dendL=dendL, dendR=dendR)


# v0.00 - classes for each cell type in network
# v0.01 - rename soma section to 'soma'
# v0.02 - add dendrite
