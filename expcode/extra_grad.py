import numpy as np
import numba as nb


def init():
    """
    Function that returns the intial \hat{u} of the algorithm
    :return:
    """
    return


def constraints():
    """
    Function that returns the constraint operators for the structured problem
    :return: F, c
    """
    return


def feat_vec(obj, label):
    """
    Function that takes a structured object and a label and returns a vector of features
    :param obj: The structured object
    :param label: The ground truth for the structured object
    :return:
    """
    return


def breg_proj(u, s, eta, norm="l2"):
    """
    Function that computes the Bregman projection of a structured object on a set
    :param u: Structured objection on which we apply the projection
    :param s: The structured object in the dual domain
    :param eta: Scale parameter eta
    :param norm: The norm that is used to compute the projection
    :return:
    """
    # DO SOME SHIT FOR EACH NORM AND RETURN THE PROJECTION
    return


def hinge_loss(w, data_iter):
    """
    Function that returns the hinge loss over a dataset
    :param w: The parameter of the model to compute the hinge loss
    :param dat_iter: Iterator over the dataset to compute the hinge loss
    :return: The hinge loss
    """
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
    s = 0.
    u_hat = init()
    u_ = 0.
    eta = 1. / lip_const

    F, c = constraints()

    for t in range(nb_epochs):
        v_w = breg_proj(u_[:dim_w], s[:dim_w], eta)
        r_w = 0.

        for i, (obj, label) in enumerate(train_iter):
            v_z = breg_proj(u_hat[dim_w:], t * (np.dot(F[i].T, u_hat[dim_w:]) + c[i]), eta)
            u_z = breg_proj(v_z, np.dot(F[i].T, v_w) + c[i], eta)

            r_w = r_w - np.dot(F[i], v_z) + feat_vec(obj, label)
            s_w = s_w  - np.dot(F[i], u_z) + feat_vec(obj, label)

            u_ = (t * u_[:dim_w] + breg_proj(v_w, r_w, eta)) / (t + 1)

    return u_
