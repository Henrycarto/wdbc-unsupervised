from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

PURPLE, GREY, LIGHT, RED = "#53057F", "#9A9A9A", "#D9C7E6", "#C0392B"


def make_figures(res, art, out):
    out = Path(out); out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    X, y, Z, pca, pred = art["X"], art["y"], art["Z"], art["pca"], art["pred"]
    P = pca.transform(Z)[:, :2]
    sgn = np.sign(pca.components_[0].sum())
    P[:, 0] *= sgn

    # 1. PCA map: cluster assignment (fill) vs diagnosis (marker), misses ringed
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    ax[0].scatter(P[pred == 0, 0], P[pred == 0, 1], s=14, color=LIGHT, label="cluster A")
    ax[0].scatter(P[pred == 1, 0], P[pred == 1, 1], s=14, color=PURPLE, label="cluster B")
    ax[0].set_title("Clusters found without labels (k-means, k=2)", loc="left")
    ax[1].scatter(P[y == 0, 0], P[y == 0, 1], s=14, color=LIGHT, label="benign")
    ax[1].scatter(P[y == 1, 0], P[y == 1, 1], s=14, color=PURPLE, label="malignant")
    miss = (y == 1) & (pred == 0)
    ax[1].scatter(P[miss, 0], P[miss, 1], s=60, facecolors="none", edgecolors=RED, lw=1.2,
                  label=f"malignant in benign cluster ({miss.sum()})")
    ax[1].set_title("Diagnosis, revealed afterwards", loc="left")
    for a in ax:
        a.set_xlabel(f"PC1 ({res['pca']['pc1']:.0%} of variance)")
        a.legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].set_ylabel(f"PC2 ({res['pca']['pc2']:.0%})")
    fig.tight_layout(); fig.savefig(out / "pca_clusters.png", dpi=160); plt.close(fig)

    # 2. sensitivity analysis: one dot per pipeline choice
    rows = res["sensitivity_analysis"]
    lab = [f"{'log ' if r['log'] else ''}{r['space']} {r['algorithm']}" for r in rows]
    ari = np.array([r["ari"] for r in rows])
    order = np.argsort(ari)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    colors = [PURPLE if (rows[i]["log"] and rows[i]["space"] == "full" and rows[i]["algorithm"] == "kmeans")
              else GREY for i in order]
    ax.barh(range(len(order)), ari[order], color=colors)
    ax.set_yticks(range(len(order)), [lab[i] for i in order], fontsize=8)
    ax.set_xlabel("Agreement with diagnosis (adjusted Rand index)")
    ax.set_xlim(0, 1)
    ax.set_title("Same data, 18 reasonable pipelines. Purple = primary, fixed before scoring", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "sensitivity.png", dpi=160); plt.close(fig)

    # 3. dendrogram (Ward), truncated, as in the 2023 deck
    fig, ax = plt.subplots(figsize=(9, 3.6))
    dendrogram(linkage(Z, "ward"), truncate_mode="lastp", p=30, color_threshold=None,
               above_threshold_color=PURPLE, no_labels=True, ax=ax)
    ax.set_title("Ward linkage on the 30 standardised log features (last 30 merges)", loc="left")
    ax.set_ylabel("merge distance")
    fig.tight_layout(); fig.savefig(out / "dendrogram.png", dpi=160); plt.close(fig)
