import numpy as np
import random

means = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
vars = [[[1., 0.], [0., 1.]], [[0.5, 0.], [0., 0.5]], [[1., 0.5], [0.5, 1.]], [[4, 1.], [1., 0.5]]]

data = np.empty([0, 2])
for i in range(10000):
    cluster = random.randrange(0, 4)
    point = np.random.multivariate_normal(mean=means[cluster], cov=vars[cluster])

    data = np.concatenate((data, point.reshape(1, -1)), axis=0)

np.save('expcode/numerical_code/data.npy', data)
