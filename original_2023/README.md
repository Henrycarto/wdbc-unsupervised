# 2023 course exercise (unchanged)

`AAMD2.py` and `iris.csv` are from the 2023 *Advanced Analysis of Medical Data* exercise: k-means
and hierarchical clustering on the Iris dataset. A separate PCA exercise was done in MATLAB and
exists only on the slides.

Kept unchanged for reference. Known issues, all fixed in the main analysis:

- It filters on `"Iris-setosa"`, but the CSV uses `setosa`, so the per-species frames are empty.
- `sns.distplot` has been removed from current seaborn.
- `KMeans` has no `random_state`, so cluster ids and colours change between runs.
- There is no quantitative comparison with the true labels.
