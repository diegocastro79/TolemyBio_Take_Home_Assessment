"""
Utility functions for processing single-cell data.

Contains functions for:
- Cell type filtering
- Outlier removal
- Gene selection (HVG + DoRothEA targets)
- Pseudobulk aggregation
"""

import pandas as pd
import numpy as np
from scipy.sparse import issparse
import anndata
import scanpy as sc
import decoupler as dc


# ============================================================================
# CELL FILTERING
# ============================================================================

def match_cell_types(cell_type_col, reference_types):
    """
    Match cell types using substring matching.
    
    Args:
        cell_type_col: pandas Series with cell type annotations
        reference_types: List of cell type keywords to match
    
    Returns:
        Boolean mask indicating which cells match
    """
    mask = pd.Series(False, index=cell_type_col.index)
    
    for ref_type in reference_types:
        mask |= cell_type_col.str.contains(ref_type, case=False, na=False)
    
    return mask


def filter_outliers_iqr(adata, column='raw_sum', multiplier=1.5):
    """
    Filter outliers using IQR method.
    
    Args:
        adata: AnnData object
        column: Column name to filter on
        multiplier: IQR multiplier for bounds (default 1.5)
    
    Returns:
        Filtered AnnData object
    """
    Q1 = adata.obs[column].quantile(0.25)
    Q3 = adata.obs[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    mask = (adata.obs[column] >= lower_bound) & (adata.obs[column] <= upper_bound)
    
    n_removed = (~mask).sum()
    print(f"Outlier filtering on '{column}':")
    print(f"  Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"  Removed: {n_removed} cells ({100*n_removed/len(mask):.2f}%)")
    
    return adata[mask].copy()


# ============================================================================
# GENE FILTERING
# ============================================================================

def select_highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3'):
    """
    Select highly variable genes using scanpy.
    
    Args:
        adata: AnnData object
        n_top_genes: Number of top variable genes to select
        flavor: Method for HVG selection ('seurat_v3', 'seurat', 'cell_ranger')
    
    Returns:
        AnnData object with HVG annotations in .var
    """
    print(f"\nSelecting highly variable genes...")
    print(f"  Method: {flavor}")
    print(f"  Target: {n_top_genes} genes")
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor
    )
    
    n_hvg = adata.var['highly_variable'].sum()
    print(f"  Selected: {n_hvg} HVGs")
    
    return adata


def filter_by_dorothea_targets(adata, confidence_levels=['A', 'B'], 
                               gene_col='feature_name', strategy='intersection'):
    """
    Filter genes to keep only DoRothEA target genes.
    
    Args:
        adata: AnnData object (should have 'highly_variable' in .var)
        confidence_levels: DoRothEA confidence levels to include
        gene_col: Column in adata.var containing gene names
        strategy: 'intersection' (HVG ∩ targets) or 'union' (HVG ∪ targets)
    
    Returns:
        Filtered AnnData object
    """
    print(f"\nFiltering by DoRothEA targets...")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Strategy: {strategy}")
    
    # Load DoRothEA
    dorothea = dc.op.dorothea(organism="human")
    dorothea_ab = dorothea[dorothea["confidence"].isin(confidence_levels)]
    
    # Get target genes
    target_genes = set(dorothea_ab['target'].unique())
    print(f"  DoRothEA targets: {len(target_genes)}")
    
    # Create masks
    hvg_mask = adata.var.get('highly_variable', pd.Series(True, index=adata.var.index))
    is_dorothea_target = adata.var[gene_col].isin(target_genes)
    
    # Apply strategy
    if strategy == 'intersection':
        final_mask = hvg_mask & is_dorothea_target
    elif strategy == 'union':
        final_mask = hvg_mask | is_dorothea_target
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"  HVGs: {hvg_mask.sum()}")
    print(f"  DoRothEA targets in data: {is_dorothea_target.sum()}")
    print(f"  Final genes ({strategy}): {final_mask.sum()}")
    
    return adata[:, final_mask].copy(), dorothea_ab


# ============================================================================
# PSEUDOBULK AGGREGATION
# ============================================================================

def create_multidimensional_pseudobulk(adata, groupby_cols=['cell_type', 'disease'],
                                       bin_cols=['raw_sum', 'nnz', 'n_measured_obs'],
                                       n_bins_per_col=5, min_cells=10, max_cells=20):
    """
    Create multidimensional bins with sub-cohort sampling for large bins.
    
    Args:
        adata: AnnData object
        groupby_cols: Columns to group by
        bin_cols: Columns to bin by
        n_bins_per_col: Number of bins per column
        min_cells: Minimum cells per pseudobulk
        max_cells: Maximum cells per pseudobulk
    
    Returns:
        pseudobulk_adata: AnnData with pseudobulk profiles
    """
    pseudobulk_data = []
    pseudobulk_obs = []
    
    print(f"\nCreating pseudobulk samples...")
    print(f"  Grouping by: {groupby_cols}")
    print(f"  Binning by: {bin_cols} ({n_bins_per_col} bins each)")
    print(f"  Cell range: {min_cells}-{max_cells} per pseudobulk")
    
    for group_keys, group_idx in adata.obs.groupby(groupby_cols).groups.items():
        cells_in_group = adata[group_idx].copy()
        
        # Create bins for each column
        bin_labels = []
        for col in bin_cols:
            try:
                bins = pd.qcut(
                    cells_in_group.obs[col], 
                    q=n_bins_per_col, 
                    labels=False, 
                    duplicates='drop'
                )
            except ValueError:
                bins = pd.cut(
                    cells_in_group.obs[col],
                    bins=n_bins_per_col,
                    labels=False
                )
            bin_labels.append(bins)
        
        # Combine bins
        cells_in_group.obs['combined_bin'] = pd.Series(
            list(zip(*bin_labels)),
            index=cells_in_group.obs.index
        ).astype(str)
        
        # Process each combined bin
        for combined_bin in cells_in_group.obs['combined_bin'].unique():
            bin_mask = cells_in_group.obs['combined_bin'] == combined_bin
            bin_cells = cells_in_group[bin_mask]
            
            n_cells_in_bin = bin_cells.n_obs
            
            if n_cells_in_bin < min_cells:
                continue
            
            # Determine number of sub-cohorts
            if n_cells_in_bin <= max_cells:
                n_subcohorts = 1
            else:
                n_subcohorts = n_cells_in_bin // max_cells
            
            # Get cell indices
            cell_indices = np.arange(n_cells_in_bin)
            np.random.seed(42)
            np.random.shuffle(cell_indices)
            
            # Create sub-cohorts
            for subcohort_id in range(n_subcohorts):
                start_idx = subcohort_id * max_cells
                end_idx = min(start_idx + max_cells, n_cells_in_bin)
                subcohort_indices = cell_indices[start_idx:end_idx]
                
                subcohort_cells = bin_cells[subcohort_indices]
                n_cells_subcohort = subcohort_cells.n_obs
                
                if n_cells_subcohort < min_cells:
                    continue
                
                # Aggregate expression
                if issparse(subcohort_cells.X):
                    pseudobulk_expr = np.array(subcohort_cells.X.sum(axis=0)).flatten()
                else:
                    pseudobulk_expr = subcohort_cells.X.sum(axis=0)
                
                pseudobulk_data.append(pseudobulk_expr)
                
                obs_dict = {
                    col: key for col, key in 
                    zip(groupby_cols, group_keys if isinstance(group_keys, tuple) else [group_keys])
                }
                obs_dict.update({
                    'combined_bin': combined_bin,
                    'subcohort_id': subcohort_id,
                    'subcohort_label': f'{combined_bin}_sub{subcohort_id}',
                    'n_cells': n_cells_subcohort,
                    'n_cells_in_bin': n_cells_in_bin
                })
                
                # Add mean values for binning columns
                for col in bin_cols:
                    obs_dict[f'{col}_mean'] = subcohort_cells.obs[col].mean()
                
                pseudobulk_obs.append(obs_dict)
    
    pseudobulk_adata = anndata.AnnData(
        X=np.array(pseudobulk_data),
        obs=pd.DataFrame(pseudobulk_obs),
        var=adata.var.copy()
    )
    
    print(f"  Created {pseudobulk_adata.n_obs} pseudobulk samples")
    
    return pseudobulk_adata
