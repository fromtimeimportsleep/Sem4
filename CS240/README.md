# Common Errors/Doubts

## Lab 0

- In Q2, using ```np.linalg.eig()``` will give the eigenvectors as column vectors of the matrix, not row vectors

## Lab 1

- In Q1 simplex tableau method, while deciding which row is the pivot. We calculate fractions and choose the smallest positive fraction, not the smallest fraction as a whole

- In Q2, it's import to pass inputs ```cvxopt.matrix``` and not ```np.matrix``` to ```solvers.qp```

## Lab 2

- In Q2, the degree with the minimum ```l2_test_loss``` was deemed as the optimal degree. It was not decided based on ```l2_train_loss``` because training loss always decreases with an increase in degree. Hence training loss was not accounted for optimal degree selection. Also, the optimal degree shouldn't be too big.
