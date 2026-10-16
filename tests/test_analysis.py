import numpy as np

from wdbc.analysis import cluster, load, preprocess, score_against_diagnosis


def test_dataset_shape():
    X, y, names = load()
    assert X.shape == (569, 30) and y.sum() == 212 and len(names) == 30


def test_clustering_never_sees_labels():
    X, y, _ = load()
    Z = preprocess(X)
    a = cluster(Z, "kmeans")
    b = cluster(Z, "kmeans")                      # same input, labels shuffled elsewhere
    np.random.default_rng(1).shuffle(y)
    c = cluster(Z, "kmeans")
    assert (a == b).all() and (a == c).all()


def test_score_mapping_and_counts():
    y = np.array([0, 0, 0, 1, 1, 1])
    lab = np.array([5, 5, 7, 7, 7, 5])            # arbitrary cluster ids
    s, pred = score_against_diagnosis(lab, y)
    assert pred.tolist() == [0, 0, 1, 1, 1, 0]
    assert s["missed_malignant"] == 1 and s["benign_flagged"] == 1


def test_primary_result_reproduces():
    X, y, _ = load()
    s, _ = score_against_diagnosis(cluster(preprocess(X), "kmeans", 2), y)
    assert s["accuracy"] == 0.921 and s["missed_malignant"] == 32
