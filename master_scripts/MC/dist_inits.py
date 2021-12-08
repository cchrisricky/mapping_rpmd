#Distributes the initial MC results (from MCfirst.py) to each Nslices many MC job
#can be run in command line on head node, it's quick
import numpy as np
import sys
import os
home = os.path.expanduser('~')
import json

with open('params.json') as f: params = json.loads( f.read() )

nbds = params['Nbds']
nlvl = params['Nlvl']
ind = 0
fmt_str = '%20.8e'
try: os.mkdir('runs')
except: pass
Nslices = params['Nslices'] #total number of slices/MC runs
nucinits = np.loadtxt('inits/nucR.dat')
eposinits = np.loadtxt('inits/mapR.dat')
emominits = np.loadtxt('inits/mapP.dat')
choices = np.random.choice( np.arange(nucinits.shape[0]), size=Nslices ) #picks Nslices many random indices of initial MC results

for i in choices:
    print(format(ind, '03d'))
    try: os.mkdir(f'runs/{ind:03d}')
    except: pass
    np.savetxt(f'runs/{ind:03d}/init_nucR.dat', nucinits[i, 1:].reshape([nbds, 1]), fmt=fmt_str)
    np.savetxt(f'runs/{ind:03d}/init_mapR.dat', eposinits[i, 1:].reshape([nbds, nlvl]), fmt=fmt_str)
    np.savetxt(f'runs/{ind:03d}/init_mapP.dat', emominits[i, 1:].reshape([nbds, nlvl]), fmt=fmt_str)
    ind += 1

print('done')