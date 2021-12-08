#Automatically randomly samples from the correct MC runs and places the initial conditions into the MD folders
#cannot be run on the head node, too slow and takes too much memory. use runinit.pbs to run
#need to follow naming convention for primary folder to work (although you can change this): Model{Nmodel}-{Nbds}Bead
import numpy as np
import sys
import os
home = os.path.expanduser('~')
top_folder = home + '/p-jkretchmer3-0' #the folder where the mod-mvrpmd, mvrpmd, and nrpmd folders live relative to home directory ~
import json

with open('params.json') as f: params = json.loads( f.read() )

rng = np.random.default_rng()
ind = 0 #starting index for files/folders
try: os.mkdir('runs')
except: pass
mcType = params['mcType'] #mc method for initial results
Nbds = params['Nbds'] #number of beads
Nlvl = params['Nlvl'] #number of electronic levels
Nmodel = params['Nmodel'] #model number
Nruns = params['Nruns'] #total number of dynamics runs
Nslices = params['Nslices'] #total number of monte carlo runs
Ntrajs = params['Ntrajs'] #number of trajectories to run per dynamic run

if Nruns > Nslices:
    #Need to solve: (A + 1)x + Ay = Nruns and x + y = Nslices to determine how many times each MC slice is sampled
    #x many MC runs will be sampled A + 1 times while y (or Nslices - x) many MC runs will be sampled A times
    A = Nruns % Nslices
    x = Nruns - A * Nslices

    #x * A + 1 samples
    for i in range(x):
        nucinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/nucR.dat')[:, 1:]
        eposinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapR.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        emominits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapP.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        outs = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/output.dat')
        samples = rng.choice(nucinits.shape[0], size=[A + 1, Ntrajs], replace=False)
        for j in range(A + 1):
            print(format(ind, '03d'))
            sys.stdout.flush()
            try: os.mkdir(f'runs/{ind:03d}')
            except: pass
            np.save(f'runs/{ind:03d}/init_nucR.npy', nucinits[ samples[j] ])
            np.save(f'runs/{ind:03d}/init_mapR.npy', eposinits[ samples[j] ])
            np.save(f'runs/{ind:03d}/init_mapP.npy', emominits[ samples[j] ])
            np.save(f'runs/{ind:03d}/init_outs.npy', outs[ samples[j] ])
            ind += 1

    #y * A samples
    for i in range(x, Nslices):
        nucinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/nucR.dat')[:, 1:]
        eposinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapR.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        emominits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapP.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        outs = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/output.dat')
        samples = rng.choice(nucinits.shape[0], size=[A, Ntrajs], replace=False)
        for j in range(A):
            print(format(ind, '03d'))
            sys.stdout.flush()
            try: os.mkdir(f'runs/{ind:03d}')
            except: pass
            np.save(f'runs/{ind:03d}/init_nucR.npy', nucinits[ samples[j] ])
            np.save(f'runs/{ind:03d}/init_mapR.npy', eposinits[ samples[j] ])
            np.save(f'runs/{ind:03d}/init_mapP.npy', emominits[ samples[j] ])
            np.save(f'runs/{ind:03d}/init_outs.npy', outs[ samples[j] ])
            ind += 1

elif Nslices > Nruns:
    #can just sample from Nruns random MC runs
    slice_samps = rng.choice(Nslices, size=Nruns, replace=False)
    for i in slice_samps:
        nucinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/nucR.dat')[:, 1:]
        eposinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapR.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        emominits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapP.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        outs = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/output.dat')
        samples = rng.choice(nucinits.shape[0], size=Ntrajs, replace=False)

        print(format(ind, '03d'))
        sys.stdout.flush()
        try: os.mkdir(f'runs/{ind:03d}')
        except: pass
        np.save(f'runs/{ind:03d}/init_nucR.npy', nucinits[ samples ])
        np.save(f'runs/{ind:03d}/init_mapR.npy', eposinits[ samples ])
        np.save(f'runs/{ind:03d}/init_mapP.npy', emominits[ samples ])
        np.save(f'runs/{ind:03d}/init_outs.npy', outs[ samples ])
        ind += 1

else:
    #Nslices = Nruns so each MC runs goes to exactly 1 MD run
    for i in range(Nslices):
        nucinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/nucR.dat')[:, 1:]
        eposinits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapR.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        emominits = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/mapP.dat')[:, 1:].reshape([-1, Nbds, Nlvl])
        outs = np.loadtxt(f'{top_folder}/{mcType}rpmd/us/MCruns/Model{Nmodel}-{Nbds}Bead/runs/{i:03d}/output.dat')
        samples = rng.choice(nucinits.shape[0], size=Ntrajs, replace=False)

        print(format(ind, '03d'))
        sys.stdout.flush()
        try: os.mkdir(f'runs/{ind:03d}')
        except: pass
        np.save(f'runs/{ind:03d}/init_nucR.npy', nucinits[ samples ])
        np.save(f'runs/{ind:03d}/init_mapR.npy', eposinits[ samples ])
        np.save(f'runs/{ind:03d}/init_mapP.npy', emominits[ samples ])
        np.save(f'runs/{ind:03d}/init_outs.npy', outs[ samples ])
        ind += 1

print('done')
