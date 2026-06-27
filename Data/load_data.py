"""
Data loading module.

Loads single-cell data from CELLxGENE Census.
"""

import cellxgene_census


def load_data(dataset_id='d18736c3-6292-4379-919a-d6d973204c87', 
              census_version="2025-11-08"):
    """
    Load data from CELLxGENE Census.
    
    Args:
        dataset_id: CELLxGENE dataset identifier
        census_version: Census version to use
    
    Returns:
        AnnData object with single-cell data
    """
    print(f"Loading data from CELLxGENE Census...")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Census version: {census_version}")
    
    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="homo_sapiens",
            obs_value_filter=f"dataset_id == '{dataset_id}'",
        )
    
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    return adata


if __name__ == "__main__":
    # Simple test of data loading
    adata = load_data()
    
    print("\nData overview:")
    print(f"  Shape: {adata.shape}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"\nTop 5 cell types:")
    for ct, count in adata.obs['cell_type'].value_counts().head(5).items():
        print(f"    {ct}: {count:,}")
