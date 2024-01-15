# Common Errors/Doubts

## Lab 0

- In Q2, using ```np.linalg.eig()``` will give the eigenvectors as column vectors of the matrix, not row vectors

## Lab 1

- In Q1 simplex tableau method, while deciding which row is the pivot. We calculate fractions and choose the smallest positive fraction, not the smallest fraction as a whole

- In Q2, it's import to pass inputs ```cvxopt.matrix``` and not ```np.matrix``` to ```solvers.qp```
