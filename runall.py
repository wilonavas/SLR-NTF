import numpy as np
import scipy.io as sio 
import math
import matplotlib.pyplot as plt
from ntfmodel import *
from utils import *

##########################################################################-----
# Load hsi in matlab format
datapath = 'data/'
filenames = ['h01-samson', 'h02-jasper','h03-urban','h03-urban6']
trials = range(10)

parms = LrModelParameters()
parms.lrate = 0.001
parms.MaxDelta = 1e-8

AbundanceThreshold = 0.95


for fn in filenames:
    # Load input dataset with input HSI and 
    # ground truth  data
    matdict = sio.loadmat(datapath + fn)
    Yin = matdict['hsiten']
    Sgt = matdict['Sgt']
    Sname = matdict['Sname']
    # plt.imshow(hsi2rgb(Yin))
    # plt.show()

    # Normalize Integers to float
    Ymax = np.max(Yin)
    Yn = Yin/Ymax
    # Get Max Intensity of each
    Ynorm = np.linalg.norm(Yn, ord=np.inf, axis=2, keepdims=True)
    Y = Yn/Ynorm

    [I,J,K] = Y.shape
    [K,R] = Sgt.shape
    Lr = int( np.maximum(I,J)**2/(R*K) )
    print(fn)
    print(f'[I,J,K]=>[{I},{J},{K}]   [Lr,R]=>[{Lr},{R}]')
    parms.prnt()
    # plt.imshow(hsi2rgb(Y))
    # plt.show()

    for i in trials:
        # Instanciate Model
        model = LrModel(Y,Lr,R,seed=i,parms=parms)
        results = model.run_optimizer()
        (cost_vector, delta, it, et) = results
        str1 = f'[I,J,K]=>[{I},{J},{K}]  [Lr,R]=>[{Lr},{R}] '
        str2 = f'|{cost_vector[-1]:10.3e} |{delta:10.3e} '
        str3 = f'|{it:10d} |{it/et:7.1f} |{et:5.0f}'
        print(str1 + str2 + str3)
        # plt.semilogy(cost_vector)
        # plt.show()
        
        # Compute endmembers using spatial components
        # and reconstructed tensor.  Re-apply norms on
        # the from the original HSI
        Sprime = get_endmembers(model, 
            AbundanceThreshold,
            norms=Ynorm)
        (Sprime,p) = reorder(Sprime,Sgt)
        # print(f'Reorder: {p}')
        # plot_decomposition(model,Sgt,Sprime,p)
        # plt.show()

        # Compute Fully Constrained Least Squares
        A = fcls_np(Y,Sprime,norms=Ynorm)
        Agt = read_agt(matdict)
        nx = math.floor(matdict['nx'])
        mtx = compute_metrics(i,Sgt, Sprime, A, Agt)
        # plot_abundance(Agt,A,nx)
        # plot_all(Agt,A,nx,model,Sgt,Sprime,p,Sname)
        # plt.show()


