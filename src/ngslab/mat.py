#!/usr/bin/env python3

"""

@author: shelvon
@email: shelvonzang@outlook.com

"""

import numpy as np
import scipy.sparse as scipy_sparse

from petsc4py import PETSc

# convert ngsolve matrix to PETSc matrices
def ngs2petscMatAIJ(ngs_mat):
    locmat = ngs_mat.local_mat
    eh, ew = locmat.entrysizes
    val,col,ind = locmat.CSR()
    ind = np.array(ind).astype(PETSc.IntType)
    col = np.array(col).astype(PETSc.IntType)
    apsc_loc = PETSc.Mat().createBAIJ(size=(eh*locmat.height, eh*locmat.width), bsize=eh, csr=(ind,col,val))

    return apsc_loc

# directly convert ngsolve matrix to numpy matrix
def ngs2numpyMat(ngs_mat):
    rows,cols,vals = ngs_mat.mat.COO()
    np_mat = scipy_sparse.csr_matrix((vals,(rows,cols))).todense()

    return np_mat

# investigate elements of the assembled matrices
def petscMat2numpyMat(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s
