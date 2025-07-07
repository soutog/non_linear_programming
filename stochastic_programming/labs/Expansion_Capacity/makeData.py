import numpy as np
import pandas as pd

###
def makeData(producers, consumers):

    ii, jj = producers, consumers
    I = list(range(ii)) #producers indices
    J = list(range(jj)) #consumer indices

    # computing producer positions
    pPos = np.zeros((2,ii))
    angles = np.arange(ii)*2*np.pi/ii
    pPos[0,:] = np.cos(angles)
    pPos[1,:] = np.sin(angles)
    pPos[:] = pPos*np.arange(ii)
    pPos[:] = pPos.round(2)

    # computing consumer positions

    njj = jj//(2*ii)
    cPos = np.zeros((2,2*ii*(njj+1)))
    # setting the horizontal components
    xjj = cPos[:1,:].reshape(-1,2*ii)
    xjj[:] = np.array((njj+1)*[np.linspace(-ii, ii, 2*ii)]).round(2)
    # setting the vertical components
    yjj = cPos[1:,:].reshape(-1,2*ii)
    aux = np.linspace(-ii,ii, njj+1)
    for i in range(njj+1): 
        yjj[i,:] = aux[i]

    distance = np.zeros((ii,jj))
    for i in range(ii):
        for j in range(jj):
            distance[i, j] =  np.linalg.norm(pPos[:,i]-cPos[:,j], ord=1)

    
    return distance, pPos, cPos
###
