import numpy as np
import numba as nb
import argparse
import os
import time
import matplotlib


def data_iter(dataset, label_type="hard"):
    """
    Iterator over a dataset that returns the structured object with its label
    :param dataset: The dataset over which we iterate
    :param label_type: The type of labels that are returned. Can be soft or hard labels
    :return:
    """
    dataset = np.random.permutation(dataset)
    for row in dataset:
        obj, rating = row[:args.obj_size], row[args.obj_size + 1:] # for now we ignore the class tuple type

        # obtain the counts for each label
        counts, _ = np.histogram(rating, range=(1, 3), bins=3)

        if label_type == "hard":
            label = np.zeros(3, dtype=np.float32)
            label[np.argmax(label)] = 1.
            yield obj, label
        elif label_type == "soft":
            yield obj, counts / counts.sum()
        else:
            raise ValueError("This is not a valid label type")


def init(size : int, norm="l1"):
    """
    Function that returns the intial \hat{u} of the algorithm
    :param size: The dimension of the \hat{u} vector
    :param norm: The norm for finding the inital point
    :return: The centroid of the 3-simplex
    """
    if norm == "l1":
        return np.ones(size, dtype=np.float32) / size
    else:
        raise NotImplementedError("This norm has not been implemented yet")


def constraints(obj, y : float, fp_cost=1., fn_cost=1.):
    """
    Function that returns the constraint operators for the structured problem
    :param obj: The structured object of size [obj_size,]
    :param y: The ground truth for the structured object of size [3, ]
    :param fp_cost: Cost of having a false positive
    :param fn_cost: Cost of having a false negative
    :return: F, c or F in y is None
    """
    F_i = np.empty([int(obj.shape[0] / 3 * 2), 3], dtype=np.float32)
    for i in range(3):
        y_ = np.zeros(3, dtype=np.float32)
        y_[i] = 1.
        F_i[:, i] = feat_vec(obj, y_)
    c_i = fp_cost - (fn_cost + fp_cost) * y
    return F_i, c_i


def lip_const(m : int, norm="l1"):
    """
    Function that computes the Lipschitz constant for F
    :param m: The number of examples in the dataset
    :param norm: The norm on the vector w
    :return: The Lipschitz constant L
    """
    if norm == "l1":
        # upper bound on the w part of the norm
        F_i = np.array([[1., 1., 1.], [-1., -1., -1.]], dtype=np.float32)
        norm_F = np.sum(np.abs(F_i), axis=0).max()
        norm_w = m * norm_F

        # upper bound on the z part of the norm
        norm_z = - np.sum(np.abs(- F_i.T), axis=0).max()

        # return the max between both upper bounds
        return np.maximum(norm_w, norm_z)
    else:
        raise NotImplementedError("This norm has not been implemented")


def feat_vec(obj, label):
    """
    Function that takes a structured object and a label and returns a vector of features
    :param obj: The structured object of size [obj_size,]
    :param label: The ground truth for the structured object of size [3, ]
    :return:
    """
    slice = int(obj.shape[0] / 3)
    plurality = np.argmax(label)
    if plurality == 0.:
        return obj[slice:]
    elif plurality == 1.:
        return np.concatenate((obj[:slice], obj[2 * slice:]))
    elif plurality == 2.:
        return obj[:2 * slice]
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
    Function that returns the average hinge loss over a dataset
    :param w: The parameter of the model to compute the hinge loss of size [obj_size / 3 * 2,]
    :param data_iter: Iterator over the dataset to compute the hinge loss
    :return: The hinge loss, accuracy over dataset
    """
    hinge = 0.
    acc = 0.
    m = 0
    for i, (obj, label) in enumerate(data_iter):
        true_hinge = np.matmul(w, feat_vec(obj, label))
        pred_hinge = []
        for index in range(3):
            label_ = np.zeros(3, dtype=np.float32)
            label_[index] = 1.
            pred_hinge.append(np.matmul(w, feat_vec(obj, label_)) + np.abs(label_ - label).sum())
        pred_hinge = np.stack(pred_hinge)
        if pred_hinge.argmax() == label.argmax():
            acc += 1.
        pred_hinge = pred_hinge.max()
        hinge += pred_hinge - true_hinge
        m += 1
    return hinge / m, acc / m


def train(train_iter, nb_epochs, dim_w, dim_z, L):
    """
    Function that trains the model
    :param train_iter: Iterator over the training set
    :param valid_iter: Iterator over the validation set
    :param nb_epochs: The number of epochs for which we train the model
    :param dim_w: Integer that gives the dimension of the w part of u
    :param dim_z: Integer that gives the dimension of the z part of u
    :param L: The lipschitz constant L
    :return:
    """
    s_w = np.zeros(dim_w, dtype=np.float32)
    u_hat = init(dim_w + dim_z)
    u_w = np.zeros(dim_w, dtype=np.float32)

    # find eta using the Lipschitz constant L
    eta = 1. / L

    for t in range(nb_epochs):
        v_w = breg_proj(u_hat[:dim_w], s_w, eta)
        r_w = 0.

        for i, (obj, label) in enumerate(train_iter):
            # generate constraints
            F_i, c_i = constraints(obj, label)

            start, stop = dim_w + 3 * i, dim_w + 3 * (i + 1)
            v_z = breg_proj(u_hat[start: stop], t * (np.dot(F_i.T, u_w) + c_i), eta)
            u_z = breg_proj(v_z, np.dot(F_i.T, v_w) + c_i, eta)

            r_w = r_w - np.dot(F_i, v_z) + feat_vec(obj, label)
            s_w = s_w - np.dot(F_i, u_z) + feat_vec(obj, label)

            u_w = (t * u_w + breg_proj(v_w, r_w, eta)) / (t + 1)

    return u_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_size",
                        type=int, default=32 * 32 * 3 * 3,
                        help="Size of a structured object")
    parser.add_argument("--data_dir",
                        type=str,
                        help="The directory where the data for the training and test sets is")
    parser.add_argument("--nb_epochs",
                        type=int, default=20,
                        help="The number of times we update"
                             "the parametes of the model")
    parser.add_argument("--save_path",
                        type=str,
                        default=None, help="Where we save the optimal w vector")
    parser.add_argument("--seed",
                        type=int,
                        default=None, help="The random seed for sampling. Use for reproducibility")
    args, unkown = parser.parse_known_args()

    # load the datasets
    train_data = np.load(os.path.join(args.data_dir, "train.npy"), mmap_mode='r')
    test_data = np.load(os.path.join(args.data_dir, "test.npy"), mmap_mode='r')

    # get iterators over the data
    np.random.seed(args.seed)
    train_iter = data_iter(train_data)
    test_iter = data_iter(test_data)

    # get the Lipschitz constant L
    m = train_data.shape[0]
    L = lip_const(m)

    # train the model to learn the optimal w
    start_time = time.time()
    w = train(train_iter, args.nb_epochs, int(args.obj_size / 3 * 2), 3 * m, L)
    print("Time spent training: ", time.time() - start_time)

    # compute the hinge loss over the test set
    loss, acc = hinge_loss(w, test_iter)

    print("The test error is: ", loss)
    print("The test accuracy is: ", acc)

    # save the w vector
    if args.save_path is not None:
        np.save(args.save_path, w)
