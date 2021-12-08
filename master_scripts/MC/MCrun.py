#Proper MC running script, will be copied from the primary folder (where params.json lives) into the runs/### folder and ran there
#should be in main folder with runjob.pbs
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser('~/joshmd'))
import json

with open('../../params.json') as f: params = json.loads( f.read() )

mcType = params['mcType']
if mcType == 'mod-mv': import mod_mvrpmd
elif mcType == 'mv': import mvrpmd
elif mcType == 'n': import nrpmd
else: raise ValueError('Invalid mcType in params.json')

nnuc     = params['Nnuc']
nstates  = params['Nlvl']
nbds     = params['Nbds']
beta     = params['beta']

mass     = np.full(nnuc, params['mass'])

mapR = np.loadtxt('init_mapR.dat').reshape([nbds, nstates])
mapP = np.loadtxt('init_mapP.dat').reshape([nbds, nstates])

nucR = np.loadtxt('init_nucR.dat').reshape([nbds, nnuc])
nucP = np.random.normal(scale=np.sqrt(mass[0] / nbds / beta), size=[nbds,nnuc])

#Potential terms
c         = params['c']
a         = params['a']
delta     = params['delta']
kvec      = np.array(mass)
amat      = np.zeros([nnuc,nstates])
amat[:,0] = a
amat[:,1] = -a
cvec      = np.zeros(nstates)
cvec[0]   = c
dvec      = np.array( [delta] )

potype    = 'harm_const_cpl'
potparams = [ kvec, amat, cvec, dvec ]

#Run test
fmt_str = '%20.8e'
Nsteps = int( params['Nsteps'] )
if mcType == 'n': resamp = 10000
else: resamp = None

#create model and run MC
if mcType == 'mod-mv': model = mod_mvrpmd.mod_mvrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
elif mcType == 'mv': model = mvrpmd.mvrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
elif mcType == 'n': model = nrpmd.nrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
model.run_MC( Nsteps, resamp=resamp )
