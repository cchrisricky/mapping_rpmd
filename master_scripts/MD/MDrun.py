#Dynamics running script that will be copied into each folder (like MCrun.py) 
# will run a trajectory for each initial value in the init files and combine together
# trajectory averages of the output.dat files and the inital and final states for each traj
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser('~/joshmd'))
import json

with open('../../params.json') as f: params = json.loads( f.read() )

mdType    = params['mdType']
nnuc      = params['Nnuc']
nstates   = params['Nlvl']
nbds      = params['Nbds']
beta      = params['beta']

if mdType == 'mod-mv': import mod_mvrpmd
elif mdType == 'mv': import mvrpmd
elif mdType == 'n': import nrpmd
else: raise ValueError('Invalid mdType in params.json')

mass = np.full(nnuc, params['mass'])

mapR = np.zeros([nbds,nstates])
mapP = np.zeros([nbds,nstates])

nucR = np.zeros([nbds,nnuc])
nucP = np.zeros([nbds,nnuc])

nucRinits = np.load('init_nucR.npy')
mapRinits = np.load('init_mapR.npy')
mapPinits = np.load('init_mapP.npy')
outsinits = np.load('init_outs.npy')

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

#Dynamics parameters
fmt_str = '%20.8e'
intype  = params['intype']
Nprint  = params['Nprint']
delt    = params['delt']
time    = params['time']
Nsteps  = int(time / delt)
Ntrajs  = params['Ntrajs']

#first trajectory (special to make average functions)
#initial conditions for first trajectory
mapR = mapRinits[0]
mapP = mapPinits[0]

nucR[:,0] = nucRinits[0]
nucP = np.random.normal(scale=np.sqrt(mass[0] * nbds / beta), size=[nbds,nnuc])

#create model and run dynamics
if mdType == 'mod-mv': model = mod_mvrpmd.mod_mvrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
elif mdType == 'mv': model = mvrpmd.mvrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
elif mdType == 'n': model = nrpmd.nrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
model.run_dynamics( Nsteps, Nprint, delt, intype )

#create average files
outputs = np.loadtxt('output.dat') #output fmt: time   etot   ke   pe   W   ...   wig_pop x 2   semi_pop x 2   xbar
nent = outputs.shape[0] #number of time entries in output
outlen = outputs.shape[1] #length of output parameters (varies based on method)
sums = np.zeros([nent, outlen + 2])
averages = np.zeros([nent, outlen + 2]) #average/sum fmt: time   etot   ke   pe   W   ...   wig_pop x 2   semi_pop x 2   xbar   sgnW   Cxx
sums[:, :outlen] = outputs; averages[:, :outlen] = outputs #saves outputs as current sum & average

#open combined files
fnames = ['nucR', 'nucP', 'mapR', 'mapP', 'output'] #files to combine first and last values of
comb_files = {}
for name in fnames:
    comb_files.update({ name : open(f'comb_{name}.dat', 'w') }) #opens all combined files and stores in dictonary

#try-finally block so if there's an error the combined files close properly before the program crashes
try:
    #write current run data to combined files
    for name in fnames:
        with open(f'{name}.dat') as f:
            file_lines = f.readlines() #creates list of lines in file
        comb_files[name].write( file_lines[0]  ) #write first/initial values
        comb_files[name].write( file_lines[-1] ) #write last/final values
        comb_files[name].write( '\n' ) #newline delimiter
        comb_files[name].flush() #flush buffer to write to files

    #calculate new parameters to average over
    sgnW = np.sign(outsinits[0, 2]) # take sign term from MC rather than dynamics
    sums[:, outlen - 5:outlen] *= sgnW #correction term since initialized to equlibrium distribution for: wig_pop, semi_pop, xbar
    sums[:, outlen]     = np.full(nent, sgnW);                      averages[:, outlen]     = np.full(nent, sgnW)              #sign of W term
    sums[:, outlen + 1] = sgnW * outputs[0, -1] * outputs[:, -1];   averages[:, outlen + 1] = outputs[0, -1] * outputs[:, -1]  #Cxx
    print(f'Average sign of W so far: {averages[0, outlen]:.4f}')

    #save sum and averages as new file (won't be reset by each trajectory)
    np.savetxt('outsum.dat', sums, fmt=fmt_str)
    np.savetxt('outave.dat', averages, fmt=fmt_str)

    #rest of the trajectories
    for i in range(1, Ntrajs):
        #initial conditions for current trajectory
        mapR = mapRinits[i]
        mapP = mapPinits[i]

        nucR[:,0] = nucRinits[i]
        nucP = np.random.normal(scale=np.sqrt(mass[0] * nbds / beta), size=[nbds,nnuc])

        #create model and run dynamics
        if mdType == 'mod-mv': model = mod_mvrpmd.mod_mvrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
        elif mdType == 'mv': model = mvrpmd.mvrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
        elif mdType == 'n': model = nrpmd.nrpmd( nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )
        model.run_dynamics( Nsteps, Nprint, delt, intype )

        #write current run data to combined files
        for name in fnames:
            with open(f'{name}.dat') as f:
                file_lines = f.readlines() #creates list of lines in file
            comb_files[name].write( file_lines[0]  ) #write first/initial values
            comb_files[name].write( file_lines[-1] ) #write last/final values
            comb_files[name].write( '\n' ) #newline delimiter
            comb_files[name].flush() #flush buffer to write to files

        #calculate new sums
        outputs = np.loadtxt('output.dat') #load new outputs
        sgnW = np.sign(outsinits[i, 2]) # take sign term from MC rather than dynamics
        sums[:, 1:outlen - 5]          += outputs[:, 1:outlen - 5]                 #etot, ke, pe, W, ...
        sums[:, outlen - 5:outlen]     += outputs[:, outlen - 5:] * sgnW           #wig_pop, semi_pop, xbar
        sums[:, outlen]                += sgnW                                     #sgnW
        sums[:, outlen + 1]            += sgnW * outputs[0, -1] * outputs[:, -1]   #Cxx

        #calculate new averages and save sums and averages
        #simple trajectory averages: etot, ke, pe, W, ..., sgnW
        averages[:, 1:outlen - 5]           = sums[:, 1:outlen - 5] / (i + 1)               #etot, ke, pe, W, ...
        averages[:, outlen]                 = sums[:, outlen] / (i + 1)                     #sgnW
        #eq averages: wig_pop, semi_pop, xbar, Cxx
        averages[:, outlen - 5:outlen]      = sums[:, outlen - 5:outlen] / sums[0, outlen]  #wig_pop, semi_pop, xbar
        averages[:, outlen + 1:]            = sums[:, outlen + 1:] / sums[0, outlen]        #Cxx
        print(f'Average sign of W so far: {averages[0, outlen]:.4f}')

        np.savetxt('outsum.dat', sums, fmt=fmt_str)
        np.savetxt('outave.dat', averages, fmt=fmt_str)

finally:
    for name in fnames:
        comb_files[name].close()

print('done')
