import numpy as np

prec = 6
eps = 2e-2
MPIsize = 6

print('brepCheck.py, precision = {0}, max error = {1}'.format(prec, eps))

for f in ['AA', 'PF']:
    for i in range(MPIsize):
        x = np.loadtxt('{0}coordinates{1}.dat'.format(f,i))
        x = np.round(x,6)
        x = x[np.lexsort([x[:,3], x[:,2], x[:,1]])]
        y = np.loadtxt('py{0}coordinates{1}.dat'.format(f,i))
        y = np.round(y,6)
        y = y[np.lexsort([y[:,3], y[:,2], y[:,1]])]
        print(f,i,((y-eps<=x)&(x<=y+eps)).all())

for f in ['GoCaxon', 'GoCbdend', 'GoCadend']:
    x = np.loadtxt('{0}coordinates.sorted.dat'.format(f))
    x = np.round(x,6)
    x = x[np.lexsort([x[:,3], x[:,2], x[:,1]])]
    y = np.loadtxt('{0}coordinates.sorted.dat'.format(f))
    y = np.round(y,6)
    y = y[np.lexsort([y[:,3], y[:,2], y[:,1]])]
    print(f,((y-eps<=x)&(x<=y+eps)).all())