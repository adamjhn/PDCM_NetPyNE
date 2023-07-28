'''
NetPyNE version of Potjans and Diesmann thalamocortical network with multicompartment neurons

netParams.py -- contains the network parameters (netParams object)

Modified to include concentration of Na, K, Cl and O2 using RxD.

'''

from netpyne import specs
import numpy as np
from cfg import cfg
from neuron.units import sec, mM
import math
import json


############################################################
#
#                    NETWORK PARAMETERS
#
############################################################

# Reescaling function (move to separate module?)
def Reescale(ScaleFactor, C, N_Full, w_p, f_ext, tau_syn, Inp, InpDC):
	if ScaleFactor<1.0 : 

		# This is a good approximation of the F_out param for the Balanced option "True".
		# Note for the Balanced=False option, it should be possible to calculate a better approximation.
		F_out=np.array([0.860, 2.600, 4.306, 5.396, 8.142, 8.188, 0.941, 7.3]) 
		
		#Para a nao balanceada o F_out deve ser outro. O nao balanceado. Apos corrigir conexoes testar para caso so de aleracao no InpPoiss
		Ncon=np.vstack(np.column_stack(0 for i in range(0,8)) for i in range(0,8))
		for r in range(0,8): 
			for c in range(0,8): 
				Ncon[r][c]=(np.log(1.-C[r][c])/np.log(1. -1./(N_Full[r]*N_Full[c])) ) /N_Full[r]

		w=w_p*np.column_stack(np.vstack( [1.0, -4.0] for i in range(0,8)) for i in range(0,4))
		w[0][2]=2.0*w[0][2]

		x1_all = w * Ncon * F_out
		x1_sum = np.sum(x1_all, axis=1)
		x1_ext = w_p * Inp * f_ext
		I_ext=np.column_stack(0 for i in range(0,8))
		I_ext = 0.001 * tau_syn * (
		        (1. - np.sqrt(ScaleFactor)) * x1_sum + 
		        (1. - np.sqrt(ScaleFactor)) * x1_ext)

		InpDC=np.sqrt(ScaleFactor)*InpDC*w_p*f_ext*tau_syn*0.001 #pA
		w_p=w_p/np.sqrt(ScaleFactor) #pA
		InpDC=InpDC+I_ext
		N_=[int(ScaleFactor*N) for N in N_Full]
	else:
		InpDC=InpDC*w_p*f_ext*tau_syn*0.001
		N_=N_Full	
	
	return InpDC, N_, w_p


###########################################################
#  Network Constants
###########################################################

# Frequency of external input
f_ext=8.0 # (Hz)
# Postsynaptic current time constant
tau_syn=0.5 # (ms)
# Membrane time constant
tau_m=10 # (ms)
# Other constants defined inside .mod
'''
#Absolute refractory period
tauref =2 (ms) 
#Reset potential
Vreset = -65 (mV) : -49 (mV) :
#Fixed firing threshold
Vteta  = -50 (mV)'''
# Membrane capacity
C_m=250 #pF
# Mean amplitude of the postsynaptic potential (in mV).
w_v=0.15
# Mean amplitude of the postsynaptic potential (in pA).
w_p= (((C_m) ** (-1) * tau_m * tau_syn / (
        tau_syn - tau_m) * ((tau_m / tau_syn) ** (
            - tau_m / (tau_m - tau_syn)) - (tau_m / tau_syn) ** (
            - tau_syn / (tau_m - tau_syn)))) ** (-1))
w_p=w_v*w_p #(pA)

#C probability of connection
C=np.array([[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.],
         [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.],
         [0.0077, 0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.],
         [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.],
         [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
         [0.0548, 0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.],
         [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
         [0.0364, 0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]])	
         
#Population size N
L=['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
N_Full=np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902])

# Number of Input per Layer
Inp=np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
if cfg.Balanced == False:
	InpUnb=np.array([2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850])

###########################################################
# Reescaling calculation
###########################################################
if cfg.DC == True:
	InpDC=Inp
	if cfg.Balanced== False:
		InpDC=InpUnb
else: 
	InpDC=np.zeros(8)
	InpPoiss=Inp*cfg.ScaleFactor
	if cfg.Balanced== False:
		InpPoiss=InpUnb*cfg.ScaleFactor
	
InpDC, N_, w_p = Reescale(cfg.ScaleFactor, C, N_Full, w_p, f_ext, tau_syn, Inp, InpDC)
print("Inputs", InpDC, N_, w_p)


############################################################
# NetPyNE Network Parameters (netParams)
############################################################

netParams = specs.NetParams()  # object of class NetParams to store the network parameters

netParams.delayMin_e = 1.5
netParams.ddelay = 0.5
netParams.delayMin_i = 0.75
netParams.weightMin = w_p
netParams.dweight = 0.1

netParams.scaleConnWeight = cfg.scaleConnWeight

############################################################
# Populations parameters
############################################################

# population locations
# from Schmidt et al 2018, PLoS Comp Bio, Macaque V1
netParams.sizeX = 300 # x-dimension (horizontal length) size in um
netParams.sizeY = 1470 # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = 300 # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

popDepths = [[0.08, 0.27], [0.08, 0.27], [0.27, 0.58], [0.27, 0.58], [0.58, 0.73], [0.58, 0.73], [0.73, 1.0], [0.73, 1.0]]

# create populations
for i in range(0,8):
    netParams.popParams[L[i]] = {'cellType': str(L[i]), 'numCells': int(N_[i]), 'cellModel': L[i], 
        'xRange': [-cfg.borderX[0], cfg.sizeX - cfg.borderX[1]],
        'yRange' : [-cfg.borderY[0] + popDepths[i][0] * cfg.sizeY, cfg.sizeY * popDepths[i][1] - cfg.borderY[1]],
        'zRange' : [-cfg.borderZ[0], cfg.sizeZ - cfg.borderZ[1]]}



############################################################
## Cell parameters
############################################################

best_ind_dict = json.load(open('cells/BPO_cellParams.json','r'))
if cfg.update_params:
    # Update cell parameters from BPO
    params = json.load(open('cells/CSTR_cellParams_fix.json','r'))
    for k,v in best_ind_dict.items():
        if '.' in k:
            hstr, loc = k.split(".")
            param, mech = hstr.split("_")
            for s in cfg.secmap[loc]:
                params['secs'][s]['mechs'][mech][param]=v
    json.dump(params, open('cells/CSTR_cellParams_updated.json','w'))

netParams.loadCellParams(label='MC', fileName='cells/CSTR_cellParams_updated.json')


############################################################
## Synaptic mechanism parameters
############################################################
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA synaptic mechanism
netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA synaptic mechanism


############################################################
# External input parameters
############################################################

if cfg.DC == False: # External Input as Poisson
	for r in range(0,8):
		netParams.popParams['poiss'+str(L[r])] = {
						'numCells': N_[r], 
						'cellModel': 'NetStim',
						'rate': InpPoiss[r]*f_ext,   
						'start': 0.0, 
						'noise': 1.0, 
						'delay': 0}
		
		auxConn=np.array([range(0,N_[r],1),range(0,N_[r],1)])
		netParams.connParams['poiss->'+str(L[r])] = {
			'preConds': {'pop': 'poiss'+str(L[r])},  
			'postConds': {'pop': L[r]},
			'connList': auxConn.T,   
			'weight': 'max(0, weightMin+normal(0,dweight*weightMin))',  
			'delay': 0.5} # 1 delay
			
# Thalamus Input: increased of 15Hz that lasts 10 ms
# 0.15 fires in 10 ms each 902 cells -> number of spikes = T*f*N_ = 0.15*902 -> 1 spike each N_*0.15
if cfg.TH == True:
	fth=15 #Hz
	Tth=10 #ms
	InTH=[0, 0, 93, 84, 0, 0, 47, 34]
	for r in [2,3,6,7]:
		nTH=int(np.sqrt(cfg.ScaleFactor)*InTH[r]*fth*Tth/1000)
		netParams.popParams['bkg_TH'+str(L[r])] = {'numCells': N_[r], 'cellModel': 'NetStim','rate': 2*(1000*nTH)/Tth ,  'start': 200.0, 'noise': 1.0, 'number': nTH, 'delay':0}
		auxConn=np.array([range(0,N_[r],1),range(0,N_[r],1)])
		netParams.connParams['bkg_TH->'+str(L[r])] = {
			'preConds': {'pop': 'bkg_TH'+str(L[r])},  
			'postConds': {'pop': L[r]},
			'connList': auxConn.T,   
			'weight':'max(0, weightMin +normal(0,dweight*weightMin))', 
			'synMech': 'exc', 
			'delay': 0.5} # 1 delay


############################################################
# Connectivity parameters
############################################################

for r in range(0,8):
	for c in range(0,8):
		if (c % 2) == 0:
			if c == 2 and r == 0:
				netParams.connParams[str(L[c])+'->'+str(L[r])] = { 
        			'preConds': {'pop': L[c]},                         # conditions of presyn cells
        			'postConds': {'pop': L[r]},                        # conditions of postsyn cells
					'divergence': cfg.ScaleFactor*(np.log(1.-C[r][c])/np.log(1. -1./(N_Full[r]*N_Full[c])) ) /N_Full[c],
        			'weight':'2*max(0, weightMin +normal(0,dweight*weightMin))', # synaptic weight
        			'delay':'max(0.1, delayMin_e +normal(0,ddelay*delayMin_e))',  # transmission delay (ms)
					'synMech': 'exc'
        			}
			else:
				netParams.connParams[str(L[c])+'->'+str(L[r])] = { 
        			'preConds': {'pop': L[c]},                         # conditions of presyn cells
        			'postConds': {'pop': L[r]},                        # conditions of postsyn cells
        			'divergence': cfg.ScaleFactor*(np.log(1.-C[r][c])/np.log(1. -1./(N_Full[r]*N_Full[c])) ) /N_Full[c],
        			'weight':'max(0, weightMin +normal(0,dweight*weightMin))', # synaptic weight
        			'delay':'max(0.1, delayMin_e +normal(0,ddelay*delayMin_e))',  # transmission delay (ms)
					'synMech': 'exc'
        			}                                                # synaptic mechanism
		else:
			netParams.connParams[str(L[c])+'->'+str(L[r])] = { 
        		'preConds': {'pop': L[c]},                         # conditions of presyn cells
        		'postConds': {'pop': L[r]},                        # conditions of postsyn cells
        		'divergence': cfg.ScaleFactor*(np.log(1.-C[r][c])/np.log(1. -1./(N_Full[r]*N_Full[c])) ) /N_Full[c],
        		'weight':'4*max(0, weightMin +normal(0,dweight*weightMin))', # synaptic weight
        		'delay':'max(0.1, delayMin_i +normal(0,ddelay*delayMin_i))',  # transmission delay (ms)
				'synMech': 'inh'
        		}                                                  # synaptic mechanism


############################################################
# RxD params
############################################################

### constants
e_charge =  1.60217662e-19
scale = 1e-14/e_charge
alpha = 5.3
constants = {'e_charge' : e_charge,
            'scale' : scale,
            'gnabar' : (30/1000) * scale,     # molecules/um2 ms mV ,
            'gnabar_l' : (0.0247/1000) * scale,
            'gkbar' : (25/1000) * scale,
            'gkbar_l' : (0.05/1000) * scale,
            'gclbar_l' : (0.1/1000) * scale,
            'ukcc2' : 0.3 * mM/sec ,
            'unkcc1' : 0.1 * mM/sec ,
            'alpha' : alpha,
            'epsilon_k_max' : 0.25/sec,
            'epsilon_o2' : 0.17/sec,
            'vtau' : 1/250.0,
            'g_gliamax' : 5 * mM/sec,
            'beta0' : 7.0,
            'avo' : 6.0221409*(10**23),
            'p_max' : 0.8, # * mM/sec,
            'nao_initial' : 144.0,
            'nai_initial' : 18.0,
            'gnai_initial' : 18.0,
            'gki_initial' : 80.0,
            'ko_initial' : 3.5,
            'ki_initial' : 140.0,
            'clo_initial' : 130.0,
            'cli_initial' : 6.0,
            'o2_bath' : cfg.o2_bath,
            'v_initial' : -70.0,
            'r0' : 100.0, 
            'k0' : 70.0}

# Update RxD constants from BPO
for k, v in best_ind_dict.items():
    if k in constants:
        constants[k] = v

#sodium activation 'm'
alpha_m = "(0.32 * (rxd.v + 54.0))/(1.0 - rxd.rxdmath.exp(-(rxd.v + 54.0)/4.0))"
beta_m = "(0.28 * (rxd.v + 27.0))/(rxd.rxdmath.exp((rxd.v + 27.0)/5.0) - 1.0)"
alpha_m0 =(0.32 * (constants['v_initial'] + 54.0))/(1.0 - math.exp(-(constants['v_initial'] + 54)/4.0))
beta_m0 = (0.28 * (constants['v_initial'] + 27.0))/(math.exp((constants['v_initial'] + 27.0)/5.0) - 1.0)
m_initial = alpha_m0/(beta_m0 + 1.0)

#sodium inactivation 'h'
alpha_h = "0.128 * rxd.rxdmath.exp(-(rxd.v + 50.0)/18.0)"
beta_h = "4.0/(1.0 + rxd.rxdmath.exp(-(rxd.v + 27.0)/5.0))"
alpha_h0 = 0.128 * math.exp(-(constants['v_initial'] + 50.0)/18.0)
beta_h0 = 4.0/(1.0 + math.exp(-(constants['v_initial'] + 27.0)/5.0))
h_initial = alpha_h0/(beta_h0 + 1.0)

#potassium activation 'n'
alpha_n = "(0.032 * (rxd.v + 52.0))/(1.0 - rxd.rxdmath.exp(-(rxd.v + 52.0)/5.0))"
beta_n = "0.5 * rxd.rxdmath.exp(-(rxd.v + 57.0)/40.0)"
alpha_n0 = (0.032 * (constants['v_initial'] + 52.0))/(1.0 - math.exp(-(constants['v_initial'] + 52.0)/5.0))
beta_n0 = 0.5 * math.exp(-(constants['v_initial'] + 57.0)/40.0)
n_initial = alpha_n0/(beta_n0 + 1.0)

netParams.rxdParams['constants'] = constants

### regions
regions = {}

#### ecs dimensions 
# margin = cfg.somaR
x = [0, cfg.sizeX]
y = [-cfg.sizeY, 0]
z = [0, cfg.sizeZ]

regions['ecs'] = {'extracellular' : True, 'xlo' : x[0],
                                          'xhi' : x[1],
                                          'ylo' : y[0],
                                          'yhi' : y[1],
                                          'zlo' : z[0],
                                          'zhi' : z[1],
                                          'dx' : 25,
                                          'volume_fraction' : cfg.alpha_ecs,
                                          'tortuosity' : cfg.tort_ecs}

regions['ecs_o2'] = {'extracellular' : True, 'xlo' : x[0],
                                            'xhi' : x[1],
                                            'ylo' : y[0],
                                            'yhi' : y[1],
                                            'zlo' : z[0],
                                            'zhi' : z[1],
                                            'dx' : 25,
                                            'volume_fraction' : 1.0,
                                            'tortuosity' : 1.0}
                                          
#xregions['cyt'] = {'cells': 'all', 'secs': 'all', 'nrn_region': 'i', 
#                 'geometry': {'class': 'FractionalVolume', 
#                 'args': {'volume_fraction': cfg.cyt_fraction, 'surface_fraction': 1}}}

#xregions['mem'] = {'cells' : 'all', 'secs' : 'all', 'nrn_region' : None, 'geometry' : 'membrane'}

regions['cyt'] = {'cells': L, 'secs': 'all', 'nrn_region': 'i', 
                'geometry': {'class': 'FractionalVolume', 
                'args': {'volume_fraction': cfg.cyt_fraction, 'surface_fraction': 1}}}

regions['mem'] = {'cells' : L, 'secs' : 'all', 'nrn_region' : None, 'geometry' : 'membrane'}

netParams.rxdParams['regions'] = regions

### species 
species = {}

k_init_str = 'ki_initial if isinstance(node, rxd.node.Node1D) else (%f if ((node.x3d - %f/2)**2+(node.y3d + %f/2)**2+(node.z3d - %f/2)**2 <= %f**2) else ko_initial)' % (cfg.k0, cfg.sizeX, cfg.sizeY, cfg.sizeZ, cfg.r0)
species['k'] = {'regions' : ['cyt', 'mem', 'ecs'], 'd' : 2.62, 'charge' : 1,
                'initial' : k_init_str,
                'ecs_boundary_conditions' : constants['ko_initial'], 'name' : 'k'}

species['na'] = {'regions' : ['cyt', 'mem', 'ecs'], 'd' : 1.78, 'charge' : 1,
                'initial' : 'nai_initial if isinstance(node, rxd.node.Node1D) else nao_initial',
                'ecs_boundary_conditions': constants['nao_initial'], 'name' : 'na'}

species['cl'] = {'regions' : ['cyt', 'mem', 'ecs'], 'd' : 2.1, 'charge' : -1,
                'initial' : 'cli_initial if isinstance(node, rxd.node.Node1D) else clo_initial',
                'ecs_boundary_conditions' : constants['clo_initial'], 'name' : 'cl'}

species['o2_extracellular'] = {'regions' : ['ecs_o2'], 'd' : 3.3, 'initial' : constants['o2_bath'],
                'ecs_boundary_conditions' : constants['o2_bath'], 'name' : 'o2'}

netParams.rxdParams['species'] = species

### parameters
params = {}
params['dump'] = {'regions' : ['cyt', 'ecs', 'ecs_o2'], 'name' : 'dump'}

params['ecsbc'] = {'regions' : ['ecs', 'ecs_o2'], 'name' : 'ecsbc', 'value' :
    '1 if (abs(node.x3d - ecs._xlo) < ecs._dx[0] or abs(node.x3d - ecs._xhi) < ecs._dx[0] or abs(node.y3d - ecs._ylo) < ecs._dx[1] or abs(node.y3d - ecs._yhi) < ecs._dx[1] or abs(node.z3d - ecs._zlo) < ecs._dx[2] or abs(node.z3d - ecs._zhi) < ecs._dx[2]) else 0'}

netParams.rxdParams['parameters'] = params

### states
netParams.rxdParams['states'] = {'vol_ratio' : {'regions' : ['cyt', 'ecs'], 'initial' : 1.0, 'name': 'volume'}, 
                                'mgate' : {'regions' : ['cyt', 'mem'], 'initial' : m_initial, 'name' : 'mgate'},
                                'hgate' : {'regions' : ['cyt', 'mem'], 'initial' : h_initial, 'name' : 'hgate'},
                                'ngate' : {'regions' : ['cyt', 'mem'], 'initial' : n_initial, 'name' : 'ngate'}}

### reactions
gna = "gnabar*mgate**3*hgate"
gk = "gkbar*ngate**4"
fko = "1.0 / (1.0 + rxd.rxdmath.exp(16.0 - k[ecs] / vol_ratio[ecs]))"
nkcc1A = "rxd.rxdmath.log((k[cyt] * cl[cyt] / vol_ratio[cyt]**2) / (k[ecs] * cl[ecs] / vol_ratio[ecs]**2))"
nkcc1B = "rxd.rxdmath.log((na[cyt] * cl[cyt] / vol_ratio[cyt]**2) / (na[ecs] * cl[ecs] / vol_ratio[ecs]**2))"
nkcc1 = "unkcc1 * (%s) * (%s+%s)" % (fko, nkcc1A, nkcc1B)
kcc2 = "ukcc2 * rxd.rxdmath.log((k[cyt] * cl[cyt] * vol_ratio[cyt]**2) / (k[ecs] * cl[ecs] * vol_ratio[ecs]**2))"

#Nerst equation - reversal potentials
ena = "26.64 * rxd.rxdmath.log(na[ecs]*vol_ratio[cyt]/(na[cyt]*vol_ratio[ecs]))"
ek = "26.64 * rxd.rxdmath.log(k[ecs]*vol_ratio[cyt]/(k[cyt]*vol_ratio[ecs]))"
ecl = "26.64 * rxd.rxdmath.log(cl[cyt]*vol_ratio[ecs]/(cl[ecs]*vol_ratio[cyt]))"

o2ecs = "o2_extracellular[ecs_o2]"
o2switch = "(1.0 + rxd.rxdmath.tanh(1e4 * (%s - 5e-4))) / 2.0" % (o2ecs)
p = "%s * p_max / (1.0 + rxd.rxdmath.exp((20.0 - (%s/vol_ratio[ecs]) * alpha)/3.0))" % (o2switch, o2ecs)
pumpA = "(%s / (1.0 + rxd.rxdmath.exp((25.0 - na[cyt] / vol_ratio[cyt])/3.0)))" % (p)
pumpB = "(1.0 / (1.0 + rxd.rxdmath.exp(3.5 - k[ecs] / vol_ratio[ecs])))"
pump = "(%s) * (%s)" % (pumpA, pumpB)
gliapump = "(1.0/3.0) * (%s / (1.0 + rxd.rxdmath.exp((25.0 - gnai_initial) / 3.0))) * (1.0 / (1.0 + rxd.rxdmath.exp(3.5 - k[ecs]/vol_ratio[ecs])))" % (p)
g_glia = "g_gliamax / (1.0 + rxd.rxdmath.exp(-((%s)*alpha/vol_ratio[ecs] - 2.5)/0.2))" % (o2ecs)
glia12 = "(%s) / (1.0 + rxd.rxdmath.exp((18.0 - k[ecs] / vol_ratio[ecs])/2.5))" % (g_glia)

# epsilon_k = "(epsilon_k_max/(1.0 + rxd.rxdmath.exp(-(((%s)/vol_ratio[ecs]) * alpha - 2.5)/0.2))) * (1.0/(1.0 + rxd.rxdmath.exp((-20 + ((1.0+1.0/beta0 -vol_ratio[ecs])/vol_ratio[ecs]) /2.0))))" % (o2ecs)
epsilon_kA = "(epsilon_k_max/(1.0 + rxd.rxdmath.exp(-((%s/vol_ratio[ecs]) * alpha - 2.5)/0.2)))" % (o2ecs)
epsilon_kB = "(1.0/(1.0 + rxd.rxdmath.exp((-20 + ((1.0+1.0/beta0 - vol_ratio[ecs])/vol_ratio[ecs]) /2.0))))"
epsilon_k = '%s * %s' % (epsilon_kA, epsilon_kB)


volume_scale = "1e-18 * avo * %f" % (1.0 / cfg.sa2v)

avo = 6.0221409*(10**23)
osm = "(1.1029 - 0.1029*rxd.rxdmath.exp( ( (na[ecs] + k[ecs] + cl[ecs] + 18.0)/vol_ratio[ecs] - (na[cyt] + k[cyt] + cl[cyt] + 132.0)/vol_ratio[cyt])/20.0))"
scalei = str(avo * 1e-18)
scaleo = str(avo * 1e-18)

### reactions
mcReactions = {}

## volume dynamics
mcReactions['vol_dyn'] = {'reactant' : 'vol_ratio[cyt]', 'product' : 'dump[ecs]', 
                        'rate_f' : "-1 * (%s) * vtau * ((%s) - vol_ratio[cyt])" % (scalei, osm), 
                        'membrane' : 'mem', 'custom_dynamics' : True,
                        'scale_by_area' : False}
                        
mcReactions['vol_dyn_ecs'] = {'reactant' : 'dump[cyt]', 'product' : 'vol_ratio[ecs]', 
                            'rate_f' : "-1 * (%s) * vtau * ((%s) - vol_ratio[cyt])" % (scaleo, osm), 
                            'membrane' : 'mem', 'custom_dynamics' : True, 
                            'scale_by_area' : False}

# # CURRENTS/LEAKS ----------------------------------------------------------------
# sodium (Na) current
mcReactions['na_current'] = {'reactant' : 'na[cyt]', 'product' : 'na[ecs]', 
                            'rate_f' : "%s * (rxd.v - %s )" % (gna, ena),
                            'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

# potassium (K) current
mcReactions['k_current'] = {'reactant' : 'k[cyt]', 'product' : 'k[ecs]', 
                            'rate_f' : "%s * (rxd.v - %s)" % (gk, ek), 
                            'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

# nkcc1 (Na+/K+/2Cl- cotransporter)
mcReactions['nkcc1_current1'] = {'reactant': 'cl[cyt]', 'product': 'cl[ecs]', 
                                'rate_f': "2.0 * (%s) * (%s)" % (nkcc1, volume_scale), 
                                'membrane': 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

mcReactions['nkcc1_current2'] = {'reactant': 'k[cyt]', 'product': 'k[ecs]',
                                'rate_f': "%s * %s" % (nkcc1, volume_scale), 
                                'membrane': 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

mcReactions['nkcc1_current3'] = {'reactant': 'na[cyt]', 'product': 'na[ecs]', 
                                'rate_f': "%s * %s" % (nkcc1, volume_scale), 
                                'membrane': 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

# ## kcc2 (K+/Cl- cotransporter)
mcReactions['kcc2_current1'] = {'reactant' : 'cl[cyt]', 'product': 'cl[ecs]', 
                                'rate_f': "%s * %s" % (kcc2, volume_scale), 
                                'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

mcReactions['kcc2_current2'] = {'reactant' : 'k[cyt]', 'product' : 'k[ecs]', 
                                'rate_f': "%s * %s" % (kcc2, volume_scale), 
                                'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

## sodium leak
mcReactions['na_leak'] = {'reactant' : 'na[cyt]', 'product' : 'na[ecs]', 
                        'rate_f' : "gnabar_l * (rxd.v - %s)" % (ena), 
                        'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

# ## potassium leak
mcReactions['k_leak'] = {'reactant' : 'k[cyt]', 'product' : 'k[ecs]', 
                        'rate_f' : "gkbar_l * (rxd.v - %s)" % (ek), 
                        'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

# ## chlorine (Cl) leak 
mcReactions['cl_current'] = {'reactant' : 'cl[cyt]', 'product' : 'cl[ecs]', 
                            'rate_f' : "gclbar_l * (%s - rxd.v)" % (ecl), 
                            'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True} 

# ## Na+/K+ pump current in neuron (2K+ in, 3Na+ out)
mcReactions['pump_current'] = {'reactant' : 'k[cyt]', 'product' : 'k[ecs]', 
                            'rate_f' : "(-2.0 * %s * %s)" % (pump, volume_scale), 
                            'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

mcReactions['pump_current_na'] = {'reactant' : 'na[cyt]', 'product' : 'na[ecs]', 
                                'rate_f' : "(3.0 * %s * %s)" % (pump, volume_scale), 
                                'membrane' : 'mem', 'custom_dynamics' : True, 'membrane_flux' : True}

# O2 depletrion from Na/K pump in neuron
# mcReactions['oxygen'] = {'reactant' : o2ecs, 'product' : 'dump[cyt]', 
#                         'rate_f' : "(%s) * (%s)" % (pump, volume_scale), 
#                         'membrane' : 'mem', 'custom_dynamics' : True}

netParams.rxdParams['multicompartmentReactions'] = mcReactions

#RATES--------------------------------------------------------------------------
rates = {}
## dm/dt
rates['m_gate'] = {'species' : 'mgate', 'regions' : ['cyt', 'mem'],
    'rate' : "((%s) * (1.0 - mgate)) - ((%s) * mgate)" % (alpha_m, beta_m)}

## dh/dt
rates['h_gate'] = {'species' : 'hgate', 'regions' : ['cyt', 'mem'],
    'rate' : "((%s) * (1.0 - hgate)) - ((%s) * hgate)" % (alpha_h, beta_h)}

## dn/dt
rates['n_gate'] = {'species' : 'ngate', 'regions' : ['cyt', 'mem'],
    'rate' : '((%s) * (1.0 - ngate)) - ((%s) * ngate)' % (alpha_n, beta_n)}

## diffusion
# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : 'ecsbc * (epsilon_o2 * (o2_bath - %s/vol_ratio[ecs]))' % (o2ecs)} # original 

# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '(epsilon_o2 * (o2_bath - %s/vol_ratio[ecs]))' % (o2ecs)} # o2everywhere

# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '(epsilon_o2 * (o2_bath - %s))' % (o2ecs)} # o2everywhereNoVolScale

rates['kdiff'] = {'species' : 'k[ecs]', 'regions' : ['ecs'],
    'rate' : 'ecsbc * ((%s) * (ko_initial - k[ecs]/vol_ratio[ecs]))' % (epsilon_k)}

rates['nadiff'] = {'species' : 'na[ecs]', 'regions' : ['ecs'],
    'rate' : 'ecsbc * ((%s) * (nao_initial - na[ecs]/vol_ratio[ecs]))' % (epsilon_k)}

rates['cldiff'] = {'species' : 'cl[ecs]', 'regions' : ['ecs'],
    'rate' : 'ecsbc * ((%s) * (clo_initial - cl[ecs]/vol_ratio[ecs]))' % (epsilon_k)}

## Glia K+/Na+ pump current 
rates['glia_k_current'] = {'species' : 'k[ecs]', 'regions' : ['ecs'],
    'rate' : '(-(%s) - (2.0 * (%s)))' % (glia12, gliapump)}

rates['glia_na_current'] = {'species' : 'na[ecs]', 'regions' : ['ecs'],
    'rate' : '(3.0 * (%s))' % (gliapump)}

## Glial O2 depletion 
# rates['o2_pump'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '-(%s)' % (gliapump)}

netParams.rxdParams['rates'] = rates

