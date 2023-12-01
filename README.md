# MolClusterkit
Toolkit containing different molecule clustering techniques and algorithms.

For now, two clustering methods are implemented:

- Butina-based clustering;

- Maximum common substructure (MCS)-based clustering.

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

### MCS
Example: 
``` python
from MolClusterkit import MCSClustering

smiles_list = [...]  # Your list of SMILES
mcs_cluster = MCSClustering(smiles_list, timeout=15)
smarts_arr, similarity_matrix = mcs_cluster.compute_similarity_matrix()
# Now you can cluster the molecules using any of the methods (see docs for more details):
clusters = mcs_cluster.dbscan_clustering(...)
clusters = mcs_cluster.hierarchical_clustering(...)
clusters = mcs_cluster.graph_based_clustering(...)
clusters = mcs_cluster.spectral_clustering(...)
```

Here, the timeout represents the wall-time in seconds that the algorithm will run for. If the timeout is reached, the algorithm will stop and return the current results. Default value is 15 seconds.

Note, this implementation just wraps RDKit's maximum common substructure algorithm. Check [here](https://www.rdkit.org/docs/source/rdkit.Chem.MCS.html#:~:text=The%20MCS%20algorithm,%3E%3E%3E) for more details on the timeout parameter.

The similarity scores obtained from `compute_similarity_matrix` is the fraction of atoms in the MCS over the total number of atoms in the smaller molecule. The obtained matrix is a square matrix with the similarity scores between all molecules in the dataset.

## CLI

MolClusterkit also provides a CLI for clustering molecules using the Butina algorithm. Small examples are in the modules docstrings.

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
