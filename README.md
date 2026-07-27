# MolClusterkit
Toolkit containing different molecule clustering techniques and algorithms.

Three clustering methods are implemented:

- **Butina-based clustering** — fingerprint similarity with the Butina algorithm;
- **MCS-based clustering** — Maximum Common Substructure similarity;
- **Rascal MCES-based clustering** — Maximum Common Edge Subgraph similarity via the [RASCAL algorithm](https://eprints.whiterose.ac.uk/3568/1/willets3.pdf).

All methods share a common set of clustering algorithms that operate on the computed similarity matrix, selectable through `cluster_molecules(algorithm=...)`: `"DBSCAN"`, `"Hierarchical"`, `"HierarchicalSilhouette"`, `"Spectral"` and `"GraphBased"`. `ButinaClustering` additionally offers `"Butina"` (its default), and `RascalMCES` offers its own fuzzy and non-fuzzy MCES clustering.

Every one of these returns an integer array with one cluster label per molecule, in
input order, and leaves the same array in `self.mol_clusters` — so the result can be
assigned straight to a dataframe column regardless of which algorithm produced it.
The exceptions are Rascal's own `fuzzy_mces_clustering` and `butina_mces_clustering`,
which return RDKit's groups of molecule indices, as a molecule may belong to more
than one fuzzy cluster. `ButinaClustering` also exposes its index groups as
`self.cluster_members`, where the first index of each tuple is the cluster centroid.

## Installation

`python -m pip install git+https://github.com/David-Araripe/MolClusterkit.git`

## Usage

### Butina-based clustering

```python
import pandas as pd

from MolClusterkit import ButinaClustering

df = pd.read_csv(...)  # Your dataframe
smiles_list = [...]  # Your list of SMILES
bclusterer = ButinaClustering(smiles_list)
clusters = bclusterer.cluster_molecules(dist_th=0.4)
# if you want to assign the clusters your dataframe:
df = df.assign(cluster_id = clusters)

# the shared algorithms are available on the same object:
clusters = bclusterer.cluster_molecules(algorithm="Hierarchical", t=5)
```

Unparseable SMILES raise a `ValueError` naming the offending indices, so the returned
cluster labels always line up one-to-one with the input list.

### MCS-based clustering

```python
from MolClusterkit import MCSClustering

smiles_list = [...]  # Your list of SMILES
mcs_cluster = MCSClustering(smiles_list, timeout=15)
smarts_arr, similarity_matrix = mcs_cluster.compute_similarity_matrix()
# Now you can cluster the molecules using any of the methods:
clusters = mcs_cluster.dbscan_clustering(...)
clusters = mcs_cluster.hierarchical_clustering(...)
clusters = mcs_cluster.hierarchical_silhouette_clustering(max_clusters=20)
clusters = mcs_cluster.graph_based_clustering(...)
clusters = mcs_cluster.spectral_clustering(...)
```

`compute_similarity_matrix` has to be called first here, and on `RascalMCES`: both run
a maximum common substructure search over every pair of molecules, which is slow and
needs a `similarity_metric` you should be choosing yourself, so neither will start one
behind your back. `ButinaClustering` is the deliberate exception — its matrix is
Tanimoto over fingerprints it already holds, so `cluster_molecules` just computes it
on demand.

The `timeout` parameter is the wall-time in seconds per pairwise MCS computation. If the timeout is reached, the algorithm stops and returns the current result. Default is 15 seconds. See RDKit's [MCS docs](https://www.rdkit.org/docs/source/rdkit.Chem.MCS.html) for more details.

Two similarity metrics are available (default is `"johnson"`):

- **Johnson**: `(atoms_mcs + bonds_mcs)² / ((atoms_mol1 + bonds_mol1) × (atoms_mol2 + bonds_mol2))`
- **smaller/mces**: `atoms_mcs / min(atoms_mol1, atoms_mol2)`

`hierarchical_silhouette_clustering` automatically selects the best number of clusters by maximizing the silhouette score across a range of 2 to `max_clusters`. The sweep is additionally capped at `n_molecules - 1`, since a silhouette score is undefined once every molecule sits in its own cluster; at least 3 molecules are required.

### Rascal MCES-based clustering

```python
from MolClusterkit import RascalMCES

smiles_list = [...]  # Your list of SMILES
rascal = RascalMCES(smiles_list, similarityThreshold=0.7)
smarts_arr, similarity_matrix = rascal.compute_similarity_matrix()
# Use any of the shared clustering algorithms:
clusters = rascal.hierarchical_silhouette_clustering(max_clusters=20)
# Or use Rascal's own clustering methods:
clusters = rascal.fuzzy_mces_clustering(cutoff=0.7)   # molecules can belong to multiple clusters
clusters = rascal.butina_mces_clustering(cutoff=0.7)   # non-fuzzy
```

`fuzzy_mces_clustering` and `butina_mces_clustering` let RDKit do the MCES comparisons
internally, so neither one needs `compute_similarity_matrix` to have been called.

The `similarityThreshold` parameter controls a fast pre-filter: molecule pairs with an estimated similarity below this threshold skip the full MCES computation and get a similarity of 0. This makes Rascal fast but can produce sparse similarity matrices. Lower the threshold if you need more complete coverage.

### Standalone MCS/MCES computation, without clustering

Neither `MCSClustering` nor `RascalMCES` needs a SMILES list to be constructed. Left
empty, they work as configured front-ends to RDKit's `rdFMCS` and `rdRascalMCES`:
set the comparison options once, then query any molecules you like. This is often the
quickest way to explore how the MCS options change the substructure you get back,
since the option objects and enums are wrapped behind plain keyword arguments.

```python
from MolClusterkit import MCSClustering, RascalMCES

# Configure once, use for any pair
mcs = MCSClustering(timeout=15)
smarts, score = mcs.mcs_similarity(("CCO", "CCN"))

rascal = RascalMCES(similarityThreshold=0.0)
smarts, score = rascal.mces_similarity(("CCO", "CCN"))
```

`mcs_in_many` takes any number of molecules at once, which makes comparing settings
a matter of changing one keyword:

```python
series = ["c1ccccc1C(=O)O", "c1ccccc1CC(=O)O", "c1ccccc1CCC(=O)O"]

strict = MCSClustering(ringMatchesRingOnly=True, completeRingsOnly=True, timeout=5)
loose = MCSClustering(atomCompare="CompareAny", bondCompare="CompareAny", timeout=5)

print(strict.mcs_in_many(series).smartsString)
print(loose.mcs_in_many(series).smartsString)
```

The keyword arguments are validated against the options RDKit accepts, so a
misspelled setting raises instead of being silently ignored. `atomCompare`,
`bondCompare` and `ringCompare` take the enum member *name* as a string (e.g.
`"CompareAny"`); the rest — `maximizeBonds`, `threshold`, `matchValences`,
`ringMatchesRingOnly`, `completeRingsOnly`, `matchChiralTag`, `verbose` — are passed
straight through. Unparseable SMILES raise a `ValueError` naming the offending
indices. Check `result.canceled` to see whether the search hit the timeout before
finding the true maximum.

### One-call clustering with best-per-cluster selection

`butina_based_clustering` and `mcs_based_clustering` wrap the classes above in a
single call that accepts a SMILES list or a `DataFrame`, assigns a `cluster_id`
column, and — with `pick_best=True` and a `score_col` — returns only the
highest-scoring compound of each cluster:

```python
from MolClusterkit import butina_based_clustering, mcs_based_clustering

df = pd.read_csv(...)  # columns: e.g. "smiles" and "pIC50"

# keep every compound, just add a cluster_id column
clustered = butina_based_clustering(df, smiles_col="smiles", dist_th=0.35)

# keep only the best-scoring compound per cluster, above a score cutoff
best = mcs_based_clustering(
    df,
    smiles_col="smiles",
    score_col="pIC50",
    score_cutoff=7.0,
    algorithm="DBSCAN",
    pick_best=True,
)
```

Pass `smiles_col=None` to auto-detect the SMILES column by name.
