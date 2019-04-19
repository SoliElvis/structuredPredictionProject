import numpy as np
import numba as nb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--feat_dim", type=int, default=32*32*3*3, help="Dimension of a structured object's feature vector")
args = parser.parse_args()


def data_iter(dataset):
    """
    Iterator over a dataset that returns the structured object with its label
    :param dataset: The dataset over which we iterate
    :return:
    """
    return


def init():
    """
    Function that returns the intial \hat{u} of the algorithm
    :return: The centroid of the 3-simplex
    """
    return np.array([0.33, 0.34, 0.33])


def constraints(y, fp_cost=1., fn_cost=1.):
    """
    Function that returns the constraint operators for the structured problem
    :param y: True label for the structured object of size [3, ]
    :param fp_cost: Cost of having a false positive
    :param fn_cost: Cost of having a false negative
    :return: F, c
    """
    F = np.array([[1., 1., 1.], [-1., -1., -1.]], dtype=np.float32)
    c = fp_cost - (fn_cost + fp_cost) * y
    return F, c


def feat_vec(obj, label):
    """
    Function that takes a structured object and a label and returns a vector of features
    :param obj: The structured object of size [3*feat_dim,]
    :param label: The ground truth for the structured object
    :return:
    """
    plurality = np.argmax(label)
    if plurality == 0.:
        return obj[args.feat_dim:]
    elif plurality == 1.:
        return np.concatenate((obj[:args.feat_dim], obj[2 * args.feat_dim:]))
    elif plurality == 2.:
        return obj[:2 * args.feat_dim]
    else:
        raise ValueError("Wrong number of dimensions for the label vector")


def breg_proj(u, s, eta, norm="l1"):
    """
    Function that computes the Bregman projection of a structured object on a set
    :param u: Structured objection on which we apply the projection
    :param s: The structured object in the dual domain
    :param eta: Scale parameter eta
    :param norm: The norm that is used to compute the projection
    :return: The Bregman updates
    """
    if norm == "l1":
        # compute numerically stable weighted softmax
        v_ = np.log(u) + eta * s
        max_ = np.amax(v_)
        return np.exp(v_ - max_) / np.exp(v_ - max_).sum()
    else:
        raise NotImplementedError("This norm has not been implemented yet")


def hinge_loss(w, data_iter):
    """
    Function that returns the hinge loss over a dataset
    :param w: The parameter of the model to compute the hinge loss of size [2*feat_dim,]
    :param data_iter: Iterator over the dataset to compute the hinge loss
    :return: The hinge loss
    """
    hinge = 0.
    for i, (obj, label) in data_iter:
        hinge += np.matmul(w, feat_vec(obj, label))
    return


def train(train_iter, valid_iter, nb_epochs, dim_w, dim_z, lip_const):
    """
    Function that trains the model
    :param train_iter: Iterator over the training set
    :param valid_iter: Iterator over the validation set
    :param nb_epochs: The number of epochs for which we train the model
    :param dim_w: Integer that gives the dimension of the w part of u
    :param dim_z: Integer that gives the dimension of the z part of u
    :param lip_const: Lipschitz constant L
    :return:
    """
    s_w = 0.
    u_hat = init()
    u_ = 0.
    eta = 1. / lip_const

    for t in range(nb_epochs):
        v_w = breg_proj(u_hat[:dim_w], s[:dim_w], eta)
        r_w = 0.

        for i, (obj, label) in enumerate(train_iter):
            # generate constraints
            F, c = constraints(label)

            v_z = breg_proj(u_hat[dim_w:], t * (np.dot(F[i].T, u_hat[dim_w:]) + c[i]), eta)
            u_z = breg_proj(v_z, np.dot(F[i].T, v_w) + c[i], eta)

            r_w = r_w - np.dot(F[i], v_z) + feat_vec(obj, label)
            s_w = s_w - np.dot(F[i], u_z) + feat_vec(obj, label)

            u_ = (t * u_[:dim_w] + breg_proj(v_w, r_w, eta)) / (t + 1)

    return u_


if __name__ == "__main__":
    # load the dataset
    # train the model to learn the optimal w using train()
    # compute the hinge loss over the test set
    pass
