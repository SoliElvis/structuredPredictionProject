import numpy as np
import random
import itertools

# types of clusters
means = [[0., 0.], [0., 3.], [4., 0.], [5., 5.]]
vars = [[[1., 0.], [0., 1.]], [[3., 0.], [0., 3.]], [[8., 4.], [4., 8.]], [[3, 2.], [2., 6.]]]

# types of matchings
matchings = [np.eye(3)[:, p].reshape(-1) for p in itertools.permutations([0, 1, 2])]
match_prob = [0.3, 0.2, 0.1, 0.1, 0.25, 0.05]

data = np.empty([0, 3*2])
label = np.empty([0, 9])
for i in range(10000):
    feature = np.empty([0, ])
    for j in range(3):
        # choose random point
        cluster = random.randrange(0, 4)
        point = np.random.multivariate_normal(mean=means[cluster], cov=vars[cluster])

        feature = np.append(feature, point.reshape(-1))

    # choose random matching
    id = np.random.choice(np.arange(6), p=match_prob)
    label = np.concatenate((label, matchings[id].reshape(1, -1)))
    data = np.concatenate((data, feature.reshape(1, -1)))

np.save('expcode/numerical_code/train_data.npy', data)
np.save('expcode/numerical_code/train_label.npy', label)

data = np.empty([0, 3*2])
label = np.empty([0, 9])
for i in range(10000):
    feature = np.empty([0, ])
    for j in range(3):
        # choose random point
        cluster = random.randrange(0, 4)
        point = np.random.multivariate_normal(mean=means[cluster], cov=vars[cluster])

        feature = np.append(feature, point.reshape(-1))

    # choose random matching
    id = np.random.choice(np.arange(6), p=match_prob)
    label = np.concatenate((label, matchings[id].reshape(1, -1)))
    data = np.concatenate((data, feature.reshape(1, -1)))

np.save('expcode/numerical_code/test_data.npy', data)
np.save('expcode/numerical_code/test_label.npy', label)
