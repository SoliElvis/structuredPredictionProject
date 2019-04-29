import numpy as np
import pickle
import random
from fasttext import FastVector
import scipy.optimize as opt
import time
from scipy.spatial.distance import hamming
import os
from matplotlib import pyplot as plt
import copy


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
    for j in range(len(en_sent)):
        for k in range(len(fr_sent)):
            score[j, k] = FastVector.cosine_similarity(en_dict[en_sent[j]], fr_dict[fr_sent[k]])

    # we find the ground truth. We randomize access to break ties randomly
    for j in range(len(en_sent)):
        argmax = int(score[j].argmax())
        truth[j, argmax] = 1.

    return truth.reshape(-1)


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
            # get the feature for a pair of words
            features = np.concatenate((en_dict[en_sent[j]], fr_dict[fr_sent[k]]))

            # put it all together to get feature for en/fr words
            table[j, k] = features

    return table.reshape(-1, dim).T


def phi(feature_matrix, label):
    """
    Fuction that returns the feature vector of two aligned sentences given the matching
    :param feature_matrix: Pre-computed Phi table where each column is the feature phi(x_jk) where jk is a possible edge
    :param label: The label. Matrix where the dimensions are the length of the two sentences
    :return: The feature vector \phi(x_i,y) where x_i is composed of the english and french sentences and y
    is the label. Its size is [2*embed_size,]
    """
    return np.matmul(feature_matrix, label)


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
    :param truth: The label that gives the true matching for a pair of sentences
    :param c_pos: The cost coefficient for a false positive
    :param c_neg: The cost coefficient for a false negative
    :return:
    """
    # The the c cost vector
    c = np.matmul(-w, feature_matrix)  # Size is  len(en) * len(fr)
    c = c + (c_neg + c_pos) * truth - c_pos

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


def train(en_data, fr_data, dim, lamb, nb_epochs):
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

    n = len(en_data)

    for k in range(nb_epochs):
        i = random.randrange(0, n)
        if len(en_data[i]) + len(fr_data[i]) > 60:  # REMOVE THIS LINE WHEN WE HAVE SPLIT THE SENTENCES
            continue

        # check if all word are in dictionary
        end = False
        for word in en_data[i]:
            if not word in en_dict:
                end = True
        for word in fr_data[i]:
            if not word in fr_dict:
                end=True
        if end:
            continue

        # find the ground truth. Must be a vector of dimension len(en_data[i]) * len(fr_data[i])
        truth = ground_truth(en_data[i], fr_data[i])

        # find the feature matrix for the pair of sentences
        feature_matrix = phi_table(en_data[i], fr_data[i], dim)

        # compute optimal label
        y_star = H(w, feature_matrix, en_data[i], fr_data[i], truth)

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
        if k > 0 and k % 10 == 0:
            print(en_data[i])
            print(fr_data[i])

            print("This is the machine translation:")
            print(get_matching(en_data[i], fr_data[i], y_star))

            print("This is the ground truth:")
            print(get_matching(en_data[i], fr_data[i], truth))

        losses.append(gap)

    return w, l, losses


if __name__ == "__main__":
    # load the datasets and perform split into training and test set
    dir = os.path.join(os.getcwd(), "expcode", "numerical_code")
    en_corpus = pickle.load(open(os.path.join(dir, 'english_vocab.pkl'), 'rb'))[:100]   # CHANGE THIS WHEN WE HAVE DB
    fr_corpus = pickle.load(open(os.path.join(dir, 'french_vocab.pkl'), 'rb'))[:100]  # CHANGE THIS WHEN WE HAVE DB

    # load the counts and co-occurences
    en_dict = FastVector(vector_file='/Users/williamst-arnaud/Downloads/cc.en.300.vec')
    fr_dict = FastVector(vector_file='/Users/williamst-arnaud/Downloads/cc.fr.300.vec')

    en_dict.apply_transform('/Users/williamst-arnaud/Downloads/fastText_multilingual-master/alignment_matrices/en.txt')
    fr_dict.apply_transform('/Users/williamst-arnaud/Downloads/fastText_multilingual-master/alignment_matrices/fr.txt')

    # number of items in dataset
    n = len(en_corpus)

    start = time.time()
    w, l, losses = train(en_corpus, fr_corpus, 2 * 300, lamb=1. / n, nb_epochs=5 * n)
    print(time.time() - start)

    # get graph of losses
    graph = plot_losses(losses)

    # show the graph of losses
    plt.show()
