import numpy as np
import matplotlib.pyplot as plt

import proxies.shapenet as shapenet


def get_truepositives(gt, pred):
    # calculates true positives between ground truth and prediction,
    # where both are binary arrays
    truepositives = 0
    for index in range(len(gt)):
        if gt[index] == 1 and pred[index] == 1:
            truepositives += 1
    return truepositives


def get_falsepositives(gt, pred):
    # calculates false positives between ground truth and prediction,
    # where both are binary arrays
    falsepositives = 0
    for index in range(len(gt)):
        if gt[index] == 0 and pred[index] == 1:
            falsepositives += 1
    return falsepositives


def get_falsenegatives(gt, pred):
    # calculates false negatives between ground truth and prediction,
    # where both are binary arrays
    falsenegatives = 0
    for index in range(len(gt)):
        if gt[index] == 1 and pred[index] == 0:
            falsenegatives += 1
    return falsenegatives


def get_tpfpfn(gt, pred):
    # calculates true positives, false positives, false negatives
    # between ground truth and prediction, where both are binary arrays
    # returns IN THAT ORDER
    tp = get_truepositives(gt, pred)
    fp = get_falsepositives(gt, pred)
    fn = get_falsenegatives(gt, pred)
    return tp, fp, fn


def get_binary_precision_recall(gt, pred):
    """
    calcules precision and recall.
    gt and pred should binary matrices of shape (n_samples, n_classes)
    return precision and recall of shape (n_classes,)
    """
    assert gt.shape == pred.shape

    precision = []
    recall = []
    for g, p in zip(gt.T, pred.T):
        tp, fp, fn = get_tpfpfn(g, p)
        try:
            precision.append(tp / (tp + fp))
        except ZeroDivisionError:
            precision.append(0)
        try:
            recall.append(tp / (tp + fn))
        except ZeroDivisionError:
            recall.append(0)

    return np.array(precision), np.array(recall)


def get_retreival_precision_recall(retreival_matrix, num_classes):
    """
    calculates precision and recall for topk returned.
    retreival_matrix should be integers of shape (n_samples, k + 1)
        first column is the class of the query object
        rest are the class of the topk
    return precision and recall of shape (k, n_classes)
    """

    # Force cast to int
    retreival_matrix = retreival_matrix.astype(int)

    topk = retreival_matrix.shape[1] - 1
    num_samples = retreival_matrix.shape[0]

    precision_stacker = []
    recall_stacker = []
    query_classes = retreival_matrix[:, 0]
    for k in range(1, retreival_matrix.shape[1]):
        retreived_classes = retreival_matrix[:, 1 : k + 1]

        pred_matrix = np.zeros((num_samples, num_classes))
        gt_matrix = np.zeros((num_samples, num_classes))
        for c in range(num_classes):
            pred_matrix[:, c] = np.array(
                [1 if c in set(e) else 0 for e in retreived_classes]
            )
            gt_matrix[:, c] = np.array([1 if c == e else 0 for e in query_classes])

        precision, recall = get_binary_precision_recall(gt_matrix, pred_matrix)
        precision_stacker.append(np.expand_dims(precision, axis=0))
        recall_stacker.append(np.expand_dims(recall, axis=0))

    return np.vstack(precision_stacker), np.vstack(recall_stacker)


def plot_object_features(obj, feature_type, **kwargs):

    # Get broken features
    bro_feat = shapenet.load_feature_from_object(obj, "b", feature_type, **kwargs)

    # Get complete features
    com_feat = shapenet.load_feature_from_object(obj, "c", feature_type, **kwargs)

    assert (bro_feat is not None) and (
        com_feat is not None
    ), "Features not loaded correctly for obect {}".format(obj)

    # Plot the actual broken vector
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.bar(range(bro_feat.shape[1]), bro_feat.squeeze())
    plt.title("Broken Feature")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Magnitude")

    # Plot the complete features with std
    fig.add_subplot(2, 1, 2)
    plt.bar(range(com_feat.shape[1]), com_feat.mean(axis=0), yerr=com_feat.std(axis=0))
    plt.title("Complete Features")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Magnitude")


def plot_object_feature_distances(obj, feature_type, **kwargs):

    # Get broken features
    bro_feat = shapenet.load_feature_from_object(obj, "b", feature_type, **kwargs)

    # Get complete features
    com_feat = shapenet.load_feature_from_object(obj, "c", feature_type, **kwargs)

    assert (bro_feat is not None) and (
        com_feat is not None
    ), "Features not loaded correctly for obect {}".format(obj)

    # Compute distance in feature space
    dists = np.linalg.norm(com_feat - bro_feat, axis=1)

    # Plot the vector distances
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.bar(range(dists.shape[0]), dists)
    plt.title("Feature Distances")
    plt.xlabel("Feature Compared")
    plt.ylabel("Magnitude")


def get_confusion_matrix(split_query, split_databse, correct_fn, value_fn, **kwargs):
    """
    Return the confusion matrix given by the distance between all combinations
    of query and database objects in the input split.
    """

    split = split_query + split_databse

    # Build empty matrix
    mat_size = len(split)
    confusion_mat = np.zeros((mat_size, mat_size))

    correct = []
    for i in range(mat_size):
        for j in range(mat_size):
            confusion_mat[i, j] = value_fn(i, j, split, **kwargs)

        # Emphasize the minimum of this column as the database model that would be returned
        best = np.argmin(confusion_mat[i, :])
        confusion_mat[i, best] = 0

        # Did we get it correct?
        if correct_fn(split[i], split[best]):
            correct.append(i)

    confusion_mat = confusion_mat[:, ~np.all(np.isinf(confusion_mat), axis=0)]
    confusion_mat = confusion_mat[~np.all(np.isinf(confusion_mat), axis=1), :]

    return confusion_mat, correct


def get_query_confusion_matrix(
    split_query, split_databse, correct_fn, value_fn, **kwargs
):
    """
    Return the confusion matrix but between the query and database objects.
    """

    split = split_query + split_databse

    # Build empty matrix
    confusion_mat = np.zeros((len(split_query), len(split_databse)))

    correct = []
    for i in range(len(split_query)):
        for j in range(len(split_databse)):
            confusion_mat[i, j] = value_fn(i, len(split_query) + j, split, **kwargs)

        # Emphasize the minimum of this column as the database model that would be returned
        best = np.argmin(confusion_mat[i, :])
        confusion_mat[i, best] = 0

        # Did we get it correct?
        if correct_fn(split_query[i], split_databse[best]):
            correct.append(i)

    confusion_mat = confusion_mat[:, ~np.all(np.isinf(confusion_mat), axis=0)]
    confusion_mat = confusion_mat[~np.all(np.isinf(confusion_mat), axis=1), :]

    return confusion_mat, correct
