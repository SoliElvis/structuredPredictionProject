from pystruct.datasets import load_scene
import numpy as np
import pickle
import random
from fasttext import FastVector
import scipy.optimize as opt
import time
import os
from matplotlib import pyplot as plt


def one_hot(integer, nb_classes):
    hot = np.zeros(nb_classes)
    for id, digit in enumerate(reversed(np.binary_repr(integer))):
        hot[-1 - id] = int(digit)

    return hot


def ground_truth(en_sent, fr_sent):
    """
    Function that extracts the ground truth for a pair of sentences in english and french
    :param en_sent: The the sentence in english
    :param fr_sent: The sentence in french
    :return:
    """
    # keys = set(fr_sent)

    # score matrix
    score = np.empty([len(en_sent), len(fr_sent)], dtype=np.float32)

    # label
    truth = np.zeros([len(en_sent), len(fr_sent)], dtype=np.float32)

    # we find the ground truth. We randomize access to break ties randomly


    return truth.reshape(-1)


def phi_table(feature, nb_classes):
    """
    Function that returns the feature matrix given a pair of sentences in english and french
    :param feature: Input x. Features
    :param dim: The dimension size of the feature vectors
    :return: The table of feature vectors that were computed for all pairs of words in the sentences
    """
    table = np.stack([feature for _ in range(nb_classes)], axis=1)
    return table


def phi(feature_matrix, label):
    """
    Fuction that returns the feature vector of two aligned sentences given the matching
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where jk is a possible edge
    :param label: The label. Matrix where the dimensions are the length of the two sentences
    :return: The feature vector \phi(x_i,y) where x_i is composed of the english and french sentences and y
    is the label. Its size is [2*embed_size,]
    """
    return np.matmul(feature_matrix, label)


def psi(feature, label, truth):
    """
    Function that returns the feature vector of a word given its label
    :param en: Sentence in english. List of tokens
    :param fr: Sentence in french. List of tokens
    :param feature: Pre-computed Phi table where each column is the feature phi(x_jk) where is a possible edge
    :param label: The matrix giving alignments for the matching
    :param truth: The matrix giving alignments for the ground-truth matching
    :return:
    """
    return phi(feature, truth) - phi(feature, label)


def H(w, feature_matrix, truth, c_pos=1, c_neg=3):
    """
    Function that computes the H_i to be optimized inside the BCWF algorithm. Solves a LP
    :param w: The vector w to evaluate the maximization oracle
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where is a possible edge
    :param en: english sentence
    :param fr: french sentence
    :param truth: The label that gives the true matching for a pair of sentences
    :param c_pos: The cost coefficient for a false positive
    :param c_neg: The cost coefficient for a false negative
    :return:
    """
    # The the c cost vector
    c = np.matmul(-w, feature_matrix)  # Size is  len(en) * len(fr)
    c = c + (c_neg + c_pos) * truth - c_pos

    # Get the constraints' RHS
    #b_ub = np.ones(2**dim, dtype=np.float32)

    # Matrix of eqiality constraints. Will fill with ones to make it unimodular -> gives integer solution -> use simplex
    #A_ub = np.stack([one_hot(i, nb_classes) for i in range(nb_classes)])

    # solve using the simplex method
    y = opt.linprog(c=c, bounds=(0., 1.))['x']

    return y


def L(truth, label, c_pos=1, c_neg=3):
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


def get_matching(en_sent, fr_sent, label):
    """
    Function that extracts the matching of words between the two sentences
    :param en_sent: The english sentence
    :param fr_sent: The french sentence
    :param label: The label (i.e. the matching) for the two sequences
    :return: The pairs of words that are matched
    """
    assert len(en_sent) * len(fr_sent) == len(label), print("The lengths are: ", len(en_sent), len(fr_sent), len(label))
    match = []
    for id in range(len(label)):
        id_en = id // len(fr_sent)
        id_fr = id % len(fr_sent)
        if label[id] == 1:
            match.append((en_sent[id_en], fr_sent[id_fr]))
    return match


def plot_losses(losses):
    """
    Function that plots the lossest that were computed after each iteration
    :param losses: Array of losses
    :return: The plot object
    """
    plt.title("Training loss after each iteration")
    plt.xlabel("Iterations")
    plt.ylabel("The training loss")
    plt.plot(np.arange(1, len(losses) + 1), losses)


def train(data, label, lamb, nb_epochs):
    """
    Function that trains a SVM using Block-Coordinate Frank-Wolfe method
    :param en_data: The english corpus
    :param fr_data: The french corpus
    :param dim: The dimension of the feature vectors
    :param lamb: The lambda value that is used for regularization
    :param nb_epochs: The number of epochs for which to train the model
    :return:
    """
    # list that gets losses to print
    losses = []

    # initialize the parameter w and the loss
    w = w_i = np.zeros(dim, dtype=np.float32)
    l = l_i = 0.

    n, nb_classes = label.shape

    for k in range(nb_epochs):
        i = random.randrange(0, n)

        # find the ground truth. Must be a vector of dimension len(en_data[i]) * len(fr_data[i])
        truth = label[i]

        # find the feature matrix for the pair of sentences
        feature_matrix = phi_table(data[i], nb_classes)

        # compute optimal label
        y_star = H(w, feature_matrix, truth)

        # lookahead step for parameter and corresponding loss
        w_s = 1. / (lamb * n) * psi(feature_matrix, y_star, truth)
        l_s = 1. / n * L(truth.reshape(-1), y_star)

        gap = lamb * ((w - w_s) @ w) - l + l_s
        gamma = gap / (lamb * np.linalg.norm(w - w_s) ** 2.)

        # update the parameters and the loss
        w_i_new = (1. - gamma) * w_i + gamma * w_s
        l_i_new = (1. - gamma) * l_i + gamma * l_s
        w = w + w_i_new - w_i
        l = l + l_i_new - l_i

        # get rid of the previous values for w_i and l_i
        w_i = w_i_new
        l_i = l_i_new

        # save the w vector every 100 iterations
        np.save(os.getcwd() + 'svm_params', w)

        # get example sentence pair and word alignment
        # if k > 0 and k % 10 == 0:

        losses.append(gap)

    return w, l, losses


if __name__ == "__main__":
    # load the datasets and perform split into training and test set
    scene = load_scene()
    X_train, X_test = scene['X_train'], scene['X_test']
    y_train, y_test = scene['y_train'], scene['y_test']

    # number of items in dataset
    n, dim = X_train.shape

    start = time.time()
    w, l, losses = train(X_train, y_train, lamb=1. / 100, nb_epochs=100)
    print(time.time() - start)

    # get graph of losses
    graph = plot_losses(losses)
    print(len(losses))

    # show the graph of losses
    plt.show()
