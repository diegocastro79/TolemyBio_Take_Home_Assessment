"""
Main data processing pipeline.

Workflow:
1. Load data from CELLxGENE Census
2. Filter immune/T/PBMC cells
3. Remove outliers
4. Select highly variable genes
5. Filter by DoRothEA targets
6. Create pseudobulk aggregations
7. Log-transform pseudobulk data
"""

import sys
sys.path.append('.')

import scanpy as sc
from Data.load_data import load_data
from Data.process_data_utils import (
    match_cell_types,
    filter_outliers_iqr,
    select_highly_variable_genes,
    filter_by_dorothea_targets,
    create_multidimensional_pseudobulk
)


# Configuration
ITP_CELL_TYPES = [
    'B cell', 'plasma cell', 'NK cell', 'monocyte', 'macrophage',
    'dendritic cell', 'neutrophil', 'basophil', 'eosinophil',
    'mast cell', 'T cell'
] # Immune/T/PBMC cells

N_HVG = 2000
DOROTHEA_CONFIDENCE = ['A', 'B']
GENE_STRATEGY = 'intersection'  # 'intersection' or 'union'

PSEUDOBULK_CONFIG = {
    'groupby_cols': ['cell_type', 'disease'],
    'bin_cols': ['raw_sum', 'nnz'],
    'n_bins_per_col': 5,
    'min_cells': 10,
    'max_cells': 20
}


# ============================================================================
# PIPELINE STEPS
# ============================================================================

def step1_load_data():
    """Step 1: Load data from CELLxGENE Census."""
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)
    
    adata = load_data()
    
    print(f"Loaded data:")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    
    return adata


def step2_filter_cell_types(adata):
    """Step 2: Filter for immune/T/PBMC cells."""
    print("\n" + "=" * 80)
    print("STEP 2: FILTER CELL TYPES")
    print("=" * 80)
    print(f"Target cell types: {ITP_CELL_TYPES}")
    
    immune_mask = match_cell_types(adata.obs['cell_type'], ITP_CELL_TYPES)
    adata_immune = adata[immune_mask].copy()
    
    print(f"\nRetained: {adata_immune.n_obs:,} cells ({100*adata_immune.n_obs/adata.n_obs:.1f}%)")
    print(f"\nTop cell types:")
    for ct, count in adata_immune.obs['cell_type'].value_counts().head(10).items():
        print(f"  {ct}: {count:,}")
    
    return adata_immune


def step3_remove_outliers(adata):
    """Step 3: Remove outlier cells."""
    print("\n" + "=" * 80)
    print("STEP 3: REMOVE OUTLIERS")
    print("=" * 80)
    
    adata_clean = filter_outliers_iqr(adata, column='raw_sum', multiplier=1.5)
    print(f"Retained: {adata_clean.n_obs:,} cells")
    
    return adata_clean


def step4_select_hvg(adata):
    """Step 4: Select highly variable genes."""
    print("\n" + "=" * 80)
    print("STEP 4: SELECT HIGHLY VARIABLE GENES")
    print("=" * 80)
    
    adata_hvg = select_highly_variable_genes(
        adata,
        n_top_genes=N_HVG,
        flavor='seurat_v3'
    )
    
    return adata_hvg


def step5_filter_by_dorothea(adata):
    """Step 5: Filter genes by DoRothEA targets."""
    print("\n" + "=" * 80)
    print("STEP 5: FILTER BY DOROTHEA TARGETS")
    print("=" * 80)
    
    adata_genes, dorothea_ab = filter_by_dorothea_targets(
        adata,
        confidence_levels=DOROTHEA_CONFIDENCE,
        gene_col='feature_name',
        strategy=GENE_STRATEGY
    )
    
    print(f"Final gene set: {adata_genes.n_vars:,} genes")
    
    return adata_genes, dorothea_ab


def step6_create_pseudobulk(adata):
    """Step 6: Create pseudobulk aggregations."""
    print("\n" + "=" * 80)
    print("STEP 6: CREATE PSEUDOBULK AGGREGATIONS")
    print("=" * 80)
    
    pseudobulk = create_multidimensional_pseudobulk(adata, **PSEUDOBULK_CONFIG)
    
    return pseudobulk


def step7_log_transform(pseudobulk):
    """Step 7: Log-transform pseudobulk data."""
    print("\n" + "=" * 80)
    print("STEP 7: LOG-TRANSFORM PSEUDOBULK DATA")
    print("=" * 80)
    print("Applying log1p transformation...")
    
    sc.pp.log1p(pseudobulk)
    
    print("  Transformation complete")
    print(f"  Data range: [{pseudobulk.X.min():.2f}, {pseudobulk.X.max():.2f}]")
    
    return pseudobulk


def print_final_summary(pseudobulk):
    """Print final pipeline summary."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\nData dimensions:")
    print(f"  Pseudobulk samples: {pseudobulk.n_obs:,}")
    print(f"  Genes (HVG ∩ DoRothEA targets): {pseudobulk.n_vars:,}")
    
    print(f"\nCells per pseudobulk:")
    print(f"  Mean: {pseudobulk.obs['n_cells'].mean():.1f}")
    print(f"  Std:  {pseudobulk.obs['n_cells'].std():.1f}")
    print(f"  Min:  {pseudobulk.obs['n_cells'].min()}")
    print(f"  Max:  {pseudobulk.obs['n_cells'].max()}")
    
    print(f"\nPseudobulks per cell type:")
    for ct, count in pseudobulk.obs['cell_type'].value_counts().items():
        print(f"  {ct}: {count}")
    
    print(f"\nPseudobulks per disease:")
    for disease, count in pseudobulk.obs['disease'].value_counts().items():
        print(f"  {disease}: {count}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_the_data():
    """
    Run the complete data processing pipeline.
    
    Returns:
        AnnData: Processed pseudobulk data (log-transformed)
    """
    print("=" * 80)
    print("SINGLE-CELL DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    # Execute pipeline steps
    adata = step1_load_data()
    adata_immune = step2_filter_cell_types(adata)
    adata_clean = step3_remove_outliers(adata_immune)
    adata_hvg = step4_select_hvg(adata_clean)
    adata_genes, dorothea_ab = step5_filter_by_dorothea(adata_hvg)
    pseudobulk = step6_create_pseudobulk(adata_genes)
    pseudobulk = step7_log_transform(pseudobulk)
    
    # Print summary
    print_final_summary(pseudobulk)
    
    return pseudobulk, dorothea_ab


if __name__ == "__main__":
    pseudobulk = process_the_data()
