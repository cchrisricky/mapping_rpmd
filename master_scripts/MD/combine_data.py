import numpy as np
import sys
import os
fmt_str = '%20.8e'
save_loc = 'combined'
import json

with open('params.json') as f: params = json.loads( f.read() )

try: os.mkdir(save_loc)
except: pass

print('000')
sys.stdout.flush()
outlen = np.loadtxt('runs/000/output.dat').shape[1]
sums = np.loadtxt('runs/000/outsum.dat')
aves = np.loadtxt('runs/000/outave.dat')
Ntrajs = int( sums[0, 1] / aves[0, 1] )
Nruns = params['Nruns']
Nlvl = params['Nlvl']
totsums = np.zeros( (Nruns,) + sums.shape )
totaves = np.zeros( (Nruns,) + aves.shape )
totsums[0] = sums
totaves[0] = aves

for i in range(1, Nruns):
    print(format(i, '03d'))
    sys.stdout.flush()
    sums = np.loadtxt(f'runs/{i:03d}/outsum.dat')
    aves = np.loadtxt(f'runs/{i:03d}/outave.dat')
    Ntrajs += int( sums[0, 1] / aves[0, 1] )
    totsums[i] = sums
    totaves[i] = aves
    np.savetxt(f'{save_loc}/sums.dat', np.sum(totsums, axis=0), fmt=fmt_str)

averages = np.zeros_like(aves)
totdev   = np.zeros_like(totsums)
stders   = np.zeros_like(aves)
averages[:, 0] = totsums[0, :, 0]                                                                                   #time
#simple trajectory averages: etot, ke, pe, W, sgnW
averages[:, 1:outlen - 1 - 2*Nlvl]          = np.sum(totsums[:,:, 1:outlen - 1 - 2*Nlvl], axis=0) / Ntrajs          #etot, ke, pe, W
averages[:, outlen]                         = np.sum(totsums[:,:, outlen], axis=0) / Ntrajs                         #sgnw
#eq averages: wig_pop, semi_pop, xbar, Cxx
totW = np.sum(totsums[:, 0, outlen])
averages[:, outlen - 1 - 2*Nlvl:outlen]     = np.sum(totsums[:,:, outlen - 1 - 2*Nlvl:outlen], axis=0) / totW       #wig_pop, semi_pop, xbar
averages[:, outlen + 1:]                    = np.sum(totsums[:,:, outlen + 1:], axis=0) / totW                      #Cxx, etc.
#std errors
for i in range(Nruns):
    totdev[i] = (totaves[i] - averages)**2
totdev[np.isinf(totdev)] = 0.0
stders = np.sqrt( np.sum(totdev, axis=0) / (Nruns - 1) ) / np.sqrt(Nruns)
stders[:, 0] = totsums[0, :, 0]

np.savetxt(f'{save_loc}/averages.dat', averages, fmt=fmt_str)
np.savetxt(f'{save_loc}/stders.dat', stders, fmt=fmt_str)
print('done')
sys.stdout.flush()