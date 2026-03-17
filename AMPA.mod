TITLE AMPA synapse with Na+ and K+ currents.

COMMENT
    AMPA receptor with equal Na+ and K+ permeability 
    Uses Goldman-Hodgkin-Katz (GHK) current equation to capture the non-linear
    I-V relationship and ~0mV reversal potential.
    
ENDCOMMENT

NEURON {
    POINT_PROCESS AMPA
    USEION na READ nai, nao WRITE ina
    USEION k  READ ki,  ko WRITE ik
    RANGE pmax 
    RANGE tau1, tau2 
}

UNITS {
    (mV)    = (millivolt)
    (nA)    = (nanoamp)
    (molar) = (1/liter)
    (mM)    = (millimolar)
    (pS)    = (picosiemens)
    (uS)    = (microsiemens)
    (um)    = (micrometers)
    FARADAY = (faraday)  (coulombs)
    R       = (k-mole)   (joule/degC)
}

PARAMETER {
    pmax    = 0.014157082995814165 (cm3/s)         : max permability
    : set to give subthreshold EPSPs similar to Exp2Syn
    : celcius=37, nai=18.0, nao=140.0, ki=140.0, ko=3.5
    pna     = 0.5                   : fractional Na+ permeability 
    pk      = 0.5                   : fractional K+  permeability
    tau1 = 0.1  (ms) <1e-9, 1e9>    : rise time constant
    tau2 = 10  (ms) <1e-9, 1e9>    : decay time constant
}

ASSIGNED {
    v       (mV)
    ina     (nA)
    ik      (nA)
    nai     (mM)
    nao     (mM)
    ki      (mM)
    ko      (mM)               : extracellular K+
    g       (1)
    fna     (mV)               : GHK Na+ factor
    fk      (mV)               : GHK K+  factor
    factor  (1)
    celsius (degC)
    area    (um2) 
}


STATE {
    A (1)
    B (1)
}

INITIAL {
    LOCAL tp
    if (tau1/tau2 > 0.9999) {
        tau1 = 0.9999 * tau2
    }
    if (tau1/tau2 < 1e-9) {
        tau1 = tau2 * 1e-9
    }
    A = 0
    B = 0
    tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1/factor
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = B - A
    : GHK factors
    fna = nao * ghkg(v, nai, nao, 1) 
    fk  = ko * ghkg(v, ki,  ko, 1)

    ina = pmax * g * pna * fna
    ik  = pmax * g * pk  * fk 
}


FUNCTION ghkg(v(mV), ci(mM), co(mM), z) (mV) {
    LOCAL xi, f, exi, fxi
    f = R*(celsius+273.15)/(z*(1e-3)*FARADAY)
    xi = v/f
    exi = exp(xi)
    if (fabs(xi) < 1e-4) {
        fxi = 1 - xi/2
    }else{
        fxi = xi/(exi - 1)
    }
    ghkg = f*((ci/co)*exi - 1)*fxi
}

DERIVATIVE state {
    A' = -A / tau1
    B' = -B / tau2
}

NET_RECEIVE(weight (uS)) {
    A = A + weight * factor
    B = B + weight * factor
}
