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

smiles_list = [...]  # Your list of SMILES
bclusterer = ButinaClustering(smiles_list)
bclusterer.cluster(cutoff=0.4)
df_with_clusters = bclusterer.assign_clusters_to_dataframe(
    df, score_col="score_column_name"
)
```

### MCS
``` python
from MolClusterkit import MCSClustering

smiles_list = [...]  # Your list of SMILES
mcs_cluster = MCSClustering(smiles_list)
mcs_cluster.compute_similarity_matrix()
labels = mcs_cluster.cluster_molecules(algorithm='DBSCAN')
```

For more details, check the docstrings 😅

## CLI

MolClusterkit also provides a CLI for clustering molecules using the Butina algorithm. Small examples are in the modules docstrings.

### Usage example;
```bash
# For mcs-based clustering
mcscluster --data_path <path_to_data> \
    --smiles_col touse_smiles \
    --score_col pchembl_value_median \
    --algorithm "DBSCAN" \
    --kwargs '{"eps": 0.3}' \
    --score_cutoff 7.0 \
    --score_col <e.g. pchembl_value> \
    --n_jobs 12

# For butina-based clustering
butinacluster --data_path <path_to_data> \
    --smiles_col touse_smiles \
    --score_col pchembl_value_median \
    --cutoff 0.7 \
    --score_cutoff 7.0 \
    --score_col <e.g. pchembl_value> \
    --n_jobs 12
```

By doing so, you can read both `.smi` and `.csv` files. While working with `.smi` files, options related to scores are not available. The `.smi` option will default as if file contained only a single SMILES per line.
