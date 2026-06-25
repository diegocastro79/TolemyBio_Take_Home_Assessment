import cellxgene_census

with cellxgene_census.open_soma(census_version="2025-11-08") as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism="homo_sapiens",
        obs_value_filter="dataset_id == 'd18736c3-6292-4379-919a-d6d973204c87'",
    )


if __name__ == "__main__":
    print(f"Overview of the data: {adata}\n")
    print(f"Number of cells: {adata.shape[0]}")