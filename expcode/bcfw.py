import numpy as np
import gensim
import pickle
import random
import itertools
import scipy.optimize as opt
import time
from scipy.spatial.distance import hamming
import os


def phi_table(en_sent, fr_sent, dim):
    """
    Function that returns the feature matrix given a pair of sentences in english and french
    :param en_sent: English sentence
    :param fr_sent: French sentence
    :param dim: The dimension size of the feature vectors
    :return: The table of feature vectors that were computed for all pairs of words in the sentences
    """
    table = np.empty([len(en_sent), len(fr_sent), dim])
    for j, en in enumerate(en_sent):
        for k, fr in enumerate(fr_sent):
            # extract the dice coefficient between the two words
            dice = (2 * co_count[(en, fr)]) / (en_count[en] + fr_count[fr])  # MODIFY THIS LINE WHEN WE HAVE DB

            # find the words relative position distance (abs, sqrt, square)
            dist = np.abs((j+1) / len(en_sent) - (k + 1) / len(fr_sent))
            sqrt_dist = np.sqrt(dist)
            sq_dist = np.square(dist)

            # compute the hamming distance between the strings
            str_len = min(len(en_sent), len(fr_sent))
            len_diff = max(len(en_sent), len(fr_sent)) - str_len
            str_dist = hamming(en[:str_len], fr[:str_len]) + len_diff

            # put it all together to get feature for en/fr words
            table[j, k] = np.array([dice, dist, sqrt_dist, sq_dist, str_dist])

    return table.reshape(-1, 5).T


def phi(feature_matrix, label):
    """
    Fuction that returns the feature vector of two aligned sentences given the matching
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where jk is a possible edge
    :param label: The label. Matrix where the dimensions are the length of the two sentences
    :return: The feature vector \phi(x_i,y) where x_i is composed of the english and french sentences and y
    is the label. Its size is [2*embed_size,]
    """
    return np.matmul(feature_matrix, label.reshape(-1))


def psi(feature_matrix, label, truth):
    """
    Function that returns the feature vector of a word given its label
    :param en: Sentence in english. List of tokens
    :param fr: Sentence in french. List of tokens
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where is a possible edge
    :param label: The matrix giving alignments for the matching
    :param truth: The matrix giving alignments for the ground-truth matching
    :return:
    """
    return phi(feature_matrix, truth) - phi(feature_matrix, label)


def H(w, feature_matrix, en, fr, truth, c_pos=1, c_neg=3):
    """
    Function that computes the H_i to be optimized inside the BCWF algorithm. Solves a LP
    :param w: The vector w to evaluate the maximization oracle
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where is a possible edge
    :param en: english sentence
    :param fr: french sentence
    :param truth: The matrix that gives the true matching for a pair of sentences
    :param c_pos: The cost coefficient for a false positive
    :param c_neg: The cost coefficient for a false negative
    :return:
    """
    # The the c cost vector
    c = np.matmul(-w, feature_matrix)  # Size is  len(en) * len(fr)
    c = c + (c_neg + c_pos) * truth.reshape(-1) - c_pos

    # Get the constraints' RHS
    b_ub = np.ones(len(en) + len(fr), dtype=np.float32)

    # Matrix of eqiality constraints. Will fill with ones to make it unimodular -> gives integer solution -> use simplex
    A_ub = np.zeros([len(en) + len(fr), len(en) * len(fr)], dtype=np.float32)
    print("The shape of the coefficient matrix is:", A_ub.shape)
    for k in range(len(fr)):
        for j in range(len(en)):
            A_ub[k, k + j * len(fr)] = 1.
    for j in range(len(en)):
        for k in range(len(fr)):
            A_ub[j + len(fr), k + j * len(fr)] = 1.

    # solve using the simplex method
    y = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0., 1.))['x']

    return y


def L(truth, label, c_pos=1, c_neg=3):
    """
    Function that computes the L_1 loss for label y (ground truth) and y_i the label we are testing
    :param truth: The ground truth. Matrix where the two dimenstions are the lengths of the sentences
    :param label: Label we are testing. Same shape as y
    :param c_pos: The cost coefficient for a false positive
    :parma c_neg: The cost coefficient for a false negative
    :return: The hamming distance between the two labels
    """
    loss = c_neg * truth * (1. - label) + c_pos * label * (1. - truth)
    return loss.sum()


def bcfw_svm(en_data, fr_data, dim, lamb, nb_epochs):
    """
    Function that trains a SVM using Block-Coordinate Frank-Wolfe method
    :param en_data: The english corpus
    :param fr_data: The french corpus
    :param dim: The dimension of the feature vectors
    :param lamb: The lambda value that is used for regularization
    :param nb_epochs: The number of epochs for which to train the model
    :return:
    """
    w = w_i = np.zeros(dim, dtype=np.float32)
    l = l_i = np.zeros(dim, dtype=np.float32)

    n = len(en_data)

    for k in range(nb_epochs):
        i = random.randrange(0, n)
        if len(en_data[i]) + len(fr_data[i]) > 60:  # REMOVE THIS LINE WHEN WE HAVE SPLIT THE SENTENCES
            continue

        # find the ground truth
        truth = np.zeros([len(en_data[i]), len(fr_data[i])])
        diag_align = min(len(en_data[i]), len(fr_data[i]))
        truth[:diag_align, :diag_align] = np.eye(diag_align)

        # find the feature matrix for the pair of sentences
        feature_matrix = phi_table(en_data[i], fr_data[i], dim)

        # compute optimal label
        y_star = H(w, feature_matrix, en_data[i], fr_data[i], truth)

        w_s = 1. / (lamb * n) * psi(feature_matrix, y_star, truth)
        l_s = 1. / n * L(truth.reshape(-1), y_star)

        gamma = 2 * n / (k + 2 * n)

        # update the parameters
        w_i_new = (1. - gamma) * w_i + gamma * w_s
        l_i_new = (1. - gamma) * l_i + gamma * l_s
        w = w + w_i_new - w_i
        l = l + l_i_new - l_i

        # get rid of the previous values for w_i and l_i
        w_i = w_i_new
        l_i = l_i_new

        # save the w vector every 100 iterations
        np.save(os.getcwd() + 'svm_params', w)

    return w, l


if __name__ == "__main__":
    # load the datasets and perform split into training and test set
    en_corpus = pickle.load(open(os.getcwd() + 'english_vocab.pkl', 'rb'))[:5000]   # CHANGE THIS WHEN WE HAVE DB
    fr_corpus = pickle.load(open(os.getcwd() + 'french_vocab.pkl', 'rb'))[:5000]  # CHANGE THIS WHEN WE HAVE DB

    # load the counts and co-occurences
    en_count = pickle.load(open(os.getcwd() + 'count_en_5000.pkl', 'rb'))  # CHANGE THIS WHEN WE HAVE DB
    fr_count = pickle.load(open(os.getcwd() + 'count_fr_5000.pkl', 'rb'))  # CHANGE THIS WHEN WE HAvE DB
    co_count = pickle.load(open(os.getcwd() + 'co_count_5000.pkl', 'rb'))  # CHANGE THIS WHEN WE HAVE DB

    start = time.time()
    w, _ = bcfw_svm(en_corpus, fr_corpus, 5, lamb=0.001, nb_epochs=1000)
    print(time.time() - start)
