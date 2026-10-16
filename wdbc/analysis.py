"""Does breast tumour cytology separate into benign and malignant without labels?

    python -m wdbc.analysis          # writes results/results.json and figures/

Data: Wisconsin Diagnostic Breast Cancer (569 fine-needle aspirates, 30 nuclear
features), shipped with scikit-learn.

Rule followed throughout: the diagnosis is NEVER used to fit or choose anything.
Preprocessing, number of clusters and algorithm are fixed by label-free criteria
first; the diagnosis is only revealed afterwards to score the result.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 0
ROOT = Path(__file__).resolve().parent.parent


def load():
    d = load_breast_cancer()
    X = d.data
    y = (d.target == 0).astype(int)          # 1 = malignant (sklearn codes malignant as 0)
    return X, y, list(d.feature_names)


def preprocess(X, log=True):
    """log1p tames the right skew of area/perimeter features, then z-score."""
    return StandardScaler().fit_transform(np.log1p(X) if log else X)


def cluster(Z, alg, k=2, seed=SEED):
    if alg == "kmeans":
        return KMeans(k, n_init=20, random_state=seed).fit_predict(Z)
    if alg == "ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    if alg == "gmm":
        return GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit_predict(Z)
    raise ValueError(alg)


def score_against_diagnosis(labels, y):
    """Only now look at the diagnosis: name each cluster by its majority class."""
    names = {c: int(round(y[labels == c].mean())) for c in np.unique(labels)}
    pred = np.array([names[c] for c in labels])
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(ari=round(adjusted_rand_score(y, labels), 3),
                accuracy=round((tp + tn) / len(y), 3),
                sensitivity_malignant=round(tp / (tp + fn), 3),
                specificity=round(tn / (tn + fp), 3),
                missed_malignant=int(fn), benign_flagged=int(fp)), pred


def run():
    X, y, names = load()
    Z = preprocess(X)
    res = {"n": int(len(y)), "malignant": int(y.sum()), "benign": int((1 - y).sum())}

    # 1. PCA (the 2023 course method), on standardised log features
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    load1 = pca.components_[0] * np.sign(pca.components_[0].sum())
    order = np.argsort(-np.abs(load1))
    res["pca"] = dict(pc1=round(float(evr[0]), 3), pc2=round(float(evr[1]), 3),
                      n_components_95=int(np.argmax(cum >= 0.95) + 1),
                      pc1_top_loadings={names[i]: round(float(load1[i]), 3) for i in order[:6]})

    # 2. choose k WITHOUT labels: silhouette over k = 2..8
    sil = {k: round(float(silhouette_score(Z, cluster(Z, "kmeans", k))), 3) for k in range(2, 9)}
    k_best = max(sil, key=sil.get)
    res["k_selection"] = dict(silhouette=sil, chosen_k=int(k_best))

    # 3. primary analysis, fixed in advance: k-means, chosen k, full standardised log space
    lab = cluster(Z, "kmeans", k_best)
    primary, pred = score_against_diagnosis(lab, y)
    res["primary"] = dict(method=f"k-means, k={k_best}, log1p + z-score, all 30 features", **primary)

    # 4. stability: 100 seeds and 200 bootstrap resamples
    ref = lab
    seeds = [adjusted_rand_score(ref, cluster(Z, "kmeans", k_best, s)) for s in range(100)]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(200):
        idx = rng.choice(len(y), len(y), replace=True)
        km = KMeans(k_best, n_init=10, random_state=SEED).fit(Z[idx])
        boot.append(adjusted_rand_score(ref, km.predict(Z)))
    res["stability"] = dict(seed_ari_min=round(float(min(seeds)), 3),
                            bootstrap_ari_median=round(float(np.median(boot)), 3),
                            bootstrap_ari_p5=round(float(np.percentile(boot, 5)), 3))

    # 5. what the misses look like: malignant tumours placed in the benign cluster
    missed = (y == 1) & (pred == 0)
    caught = (y == 1) & (pred == 1)
    key = ["mean radius", "mean concave points", "worst concave points", "worst area", "mean texture"]
    res["missed_vs_caught_malignant"] = {
        f: dict(missed_median=round(float(np.median(X[missed, names.index(f)])), 3),
                caught_median=round(float(np.median(X[caught, names.index(f)])), 3),
                benign_median=round(float(np.median(X[y == 0, names.index(f)])), 3))
        for f in key}

    # 6. sensitivity analysis: every other reasonable choice, reported, not cherry-picked
    sens = []
    for log, space, alg in itertools.product([True, False], ["full", "pca2", "pca95"], ["kmeans", "ward", "gmm"]):
        Zs = preprocess(X, log)
        if space == "pca2":
            Zs = PCA(2, random_state=SEED).fit_transform(Zs)
        elif space == "pca95":
            Zs = PCA(0.95, random_state=SEED).fit_transform(Zs)
        s, _ = score_against_diagnosis(cluster(Zs, alg, 2), y)
        sens.append(dict(log=log, space=space, algorithm=alg, **s))
    res["sensitivity_analysis"] = sens
    aris = [r["ari"] for r in sens]
    res["sensitivity_range"] = dict(ari_min=min(aris), ari_max=max(aris),
                                    accuracy_min=min(r["accuracy"] for r in sens),
                                    accuracy_max=max(r["accuracy"] for r in sens))

    # 7. reference ceiling: what a supervised model gets with the labels
    cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
    sup = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=5000))
    res["supervised_reference"] = dict(model="logistic regression, 10-fold CV",
                                       accuracy=round(float(cross_val_score(sup, X, y, cv=cv).mean()), 3),
                                       sensitivity=round(float(cross_val_score(sup, X, y, cv=cv, scoring="recall").mean()), 3))
    return res, dict(X=X, y=y, Z=Z, names=names, pca=pca, labels=lab, pred=pred, sil=sil)


def main():
    res, art = run()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(res, indent=2), newline="\n")
    from .figures import make_figures
    make_figures(res, art, ROOT / "figures")
    p = res["primary"]
    print(f"primary: {p['method']}")
    print(f"  ARI {p['ari']}  accuracy {p['accuracy']}  sensitivity {p['sensitivity_malignant']}  "
          f"specificity {p['specificity']}  missed {p['missed_malignant']}/{res['malignant']}")
    print(f"  stability: seed ARI min {res['stability']['seed_ari_min']}, bootstrap median {res['stability']['bootstrap_ari_median']}")
    print(f"  sensitivity analysis ARI range {res['sensitivity_range']}")
    print(f"  supervised reference {res['supervised_reference']}")
    print(f"  PCA {res['pca']}")
    print(f"  k selection {res['k_selection']}")
    print(json.dumps(res["missed_vs_caught_malignant"], indent=1))


if __name__ == "__main__":
    main()
