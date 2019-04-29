import numpy as np
from matplotlib import pyplot as plt
import argparse
import scipy.optimize as opt
import random


def phi_table(x, dim):
    """
    Function that returns the feature matrix given a pair of sentences in english and french
    :param x: Vector of size 6
    :return: The table of feature vectors that were computed for all pairs of words in the sentences
    """
    table = np.empty([3, 3, dim])

    x = x.reshape(3, 2)

    for j in range(3):
        for k in range(3):
            table[j, k] = np.abs(x[j] - x[k]) ** 2.# np.concatenate((x[j], x[k]))

    return table.reshape(-1, dim).T


def phi(x, y):
    """
    Fuction that returns the feature vector of two aligned sentences given the matching
    :param x: The input x
    :param y: The label
    :return: The feature vector \phi(x_i,y_i)
    is the label. Its size is [2*embed_size,]
    """
    return np.matmul(x, y)


def psi(feature_matrix, label, truth):
    """
    Function that returns the feature vector of a word given its label
    :param en: Sentence in english. List of tokens
    :param fr: Sentence in french. List of tokens
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where is a possible edge
    :param label: The label giving alignments for the matching
    :param truth: The label giving alignments for the ground-truth matching
    :return:
    """
    return phi(feature_matrix, truth) - phi(feature_matrix, label)


def H(w, feature_matrix, truth, c_pos=1, c_neg=3):
    """
    Function that computes the H_i to be optimized inside the BCWF algorithm. Solves a LP
    :param w: The vector w to evaluate the maximization oracle
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where is a possible edge
    :param truth: The label that gives the true matching for a pair of sentences
    :param c_pos: The cost coefficient for a false positive
    :param c_neg: The cost coefficient for a false negative
    :return:
    """
    # The the c cost vector
    c = np.matmul(-w, feature_matrix)  # Size is  len(en) * len(fr)
    c = c + (c_neg + c_pos) * truth - c_pos

    # Get the constraints' RHS
    b_ub = np.ones(6, dtype=np.float32)

    # Matrix of eqiality constraints. Will fill with ones to make it unimodular -> gives integer solution -> use simplex
    A_ub = np.zeros([6, 9], dtype=np.float32)
    # print("The shape of the coefficient matrix is:", A_ub.shape)
    for k in range(3):
        for j in range(3):
            A_ub[k, k + j * 3] = 1.
    for j in range(3):
        for k in range(3):
            A_ub[j + 3, k + j * 3] = 1.

    # solve using the simplex method
    y = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0., 1.))['x']

    return y


def L(truth, label, c_pos=1., c_neg=3.):
    """
    Function that computes the L_1 loss for label y (ground truth) and y_i the label we are testing
    :param truth: The ground truth. Vector where the two dimension is the product of the length of the two sentences
    :param label: Label we are testing. Same shape as y
    :param c_pos: The cost coefficient for a false positive
    :parma c_neg: The cost coefficient for a false negative
    :return: The hamming distance between the two labels
    """
    loss = c_neg * truth * (1. - label) + c_pos * label * (1. - truth)
    return loss.sum()


def plot_losses(losses):
    """
    Function that plots the lossest that were computed after each iteration
    :param losses: Array of losses
    :return: The plot object
    """
    plt.title("Training loss after each iteration")
    plt.xlabel("Iterations")
    plt.ylabel("The training loss")
    plt.yscale('log')
    plt.plot(np.arange(1, len(losses) + 1), losses)


def train(data, label, dim, lamb, nb_epochs):
    """
    Function that trains a SVM using Block-Coordinate Frank-Wolfe method
    :param data: The dataset on which we train
    :param label: The assignments for each x_i
    :param lamb: The lambda value that is used for regularization
    :param nb_epochs: The number of epochs for which to train the model
    :return:
    """
    # list that gets losses to print
    losses = []

    # initialize the parameter w and the loss
    w = w_i = np.zeros(dim, dtype=np.float32)
    l = l_i = 0.

    n = len(data)

    for k in range(nb_epochs):
        i = random.randrange(0, n)

        # find the ground truth. Must be a vector of dimension len(en_data[i]) * len(fr_data[i])
        truth = label[i]

        # find the feature matrix for the pair of sentences
        feature_matrix = phi_table(data[i], 2)

        # compute optimal label
        y_star = H(w, feature_matrix, truth)

        # lookahead step for parameter and corresponding loss
        w_s = (1. / (lamb * n)) * psi(feature_matrix, y_star, truth)
        l_s = (1. / n) * L(truth.reshape(-1), y_star)

        gamma = 2 * n / (k + 2 * n)

        # update the parameters and the loss
        w_i_new = (1. - gamma) * w_i + gamma * w_s
        l_i_new = (1. - gamma) * l_i + gamma * l_s
        w = w + w_i_new - w_i
        l = l + l_i_new - l_i

        # get rid of the previous values for w_i and l_i
        w_i = w_i_new
        l_i = l_i_new

        # save the w vector every 100 iterations
        np.save('svm_params.npy', w)

        # add the loss to a list to plot
        losses.append(lamb * ((w - w_s) @ w) - l + l_s)

    return w, l, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Synthetic dataset to load and train on")
    parser.add_argument("label", type=str, help="Synthetic labels for the dataset")
    args = parser.parse_args()

    train_data = np.load(args.data)
    train_label = np.load(args.label)

    n = len(train_data)

    w, l, losses = train(train_data, train_label, 2, lamb=0.01, nb_epochs=2000)

    print("Final loss is: ", l)

    plot_losses(losses)

    plt.show()
