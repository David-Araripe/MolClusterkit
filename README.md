# MolClusterkit
Toolkit containing different molecule clustering techniques and algorithms.

Three clustering methods are implemented:

- **Butina-based clustering** — fingerprint similarity with the Butina algorithm;
- **MCS-based clustering** — Maximum Common Substructure similarity;
- **Rascal MCES-based clustering** — Maximum Common Edge Subgraph similarity via the [RASCAL algorithm](https://eprints.whiterose.ac.uk/3568/1/willets3.pdf).

All methods share a common set of clustering algorithms (DBSCAN, hierarchical, spectral, graph-based) that operate on the computed similarity matrix.

## Installation

`python -m pip install git+https://github.com/David-Araripe/MolClusterkit.git`

## Usage

### Butina-based clustering

```python
from MolClusterkit import ButinaClustering

df = pd.read_csv(...)  # Your dataframe
smiles_list = [...]  # Your list of SMILES
bclusterer = ButinaClustering(smiles_list)
clusters = bclusterer.cluster_molecules(cutoff=0.4)
# if you want to assign the clusters your dataframe:
df = df.assign(cluster_id = clusters)
```

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

The `timeout` parameter is the wall-time in seconds per pairwise MCS computation. If the timeout is reached, the algorithm stops and returns the current result. Default is 15 seconds. See RDKit's [MCS docs](https://www.rdkit.org/docs/source/rdkit.Chem.MCS.html) for more details.

Two similarity metrics are available (default is `"johnson"`):

- **Johnson**: `(atoms_mcs + bonds_mcs)² / ((atoms_mol1 + bonds_mol1) × (atoms_mol2 + bonds_mol2))`
- **smaller/mces**: `atoms_mcs / min(atoms_mol1, atoms_mol2)`

`hierarchical_silhouette_clustering` automatically selects the best number of clusters by maximizing the silhouette score across a range of 2 to `max_clusters`.

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

The `similarityThreshold` parameter controls a fast pre-filter: molecule pairs with an estimated similarity below this threshold skip the full MCES computation and get a similarity of 0. This makes Rascal fast but can produce sparse similarity matrices. Lower the threshold if you need more complete coverage.

### Standalone similarity computation

Both `MCSClustering` and `RascalMCES` can be used to compute pairwise similarity without providing a full SMILES list upfront. This is useful when you want to configure the algorithm once and compute similarities for arbitrary pairs:

```python
from MolClusterkit import MCSClustering, RascalMCES

# Configure once, use for any pair
mcs = MCSClustering(timeout=15)
smarts, score = mcs.mcs_similarity(("CCO", "CCN"))

rascal = RascalMCES(similarityThreshold=0.0)
smarts, score = rascal.mces_similarity(("CCO", "CCN"))
```

## CLI

MolClusterkit also provides CLIs for Butina and MCS clustering.

### Usage example;
```bash
# For mcs-based clustering
mcscluster -i "path/to/data.csv" \              # --input_path
    -smic "SMILES" \                            # --smiles_col
    -scor "pIC50" \  # example..                # --score_col
    -cut 7.0 \                                  # --score_cutoff
    -a "DBSCAN" \                               # --algorithm
    -k '{"eps": 0.3}' \                         # --kwargs
    -j 12 \                                     # --n_jobs
    -p \                                        # --pick_best
    -to 1.5 \                                   # --timeout
    -mcs '{"AtomCompare": "CompareElements"}' \ # --mcs_kwargs
    -o "path/to/output.csv"                     # --output_path

# For butina-based clustering
butinacluster -i "path/to/data.csv" \                # --input_path
    -smic "SMILES" \                                 # --smiles_col
    -scor "pIC50" \  # example..                     # --score_col
    -cut 7.0 \                                       # --score_cutoff
    -dist 0.35 \                                     # --dist_th
    -j 12 \                                          # --n_jobs
    -p \                                             # --pick_best
    -o "path/to/output.csv"                          # --output_path
```

Both commands support calling on `.smi`, `tsv` and `.csv` files. While working with `.smi` files, options related to scores are not available. The `.smi` option will default as if file contained only a single SMILES per line.

For more information, run `mcscluster -h` or `butinacluster -h`.
