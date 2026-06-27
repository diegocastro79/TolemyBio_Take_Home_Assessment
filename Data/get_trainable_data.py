import torch
from Data.process_data import process_the_data
import pickle
from pathlib import Path

def get_network_dict(dorothea_ab, selected_genes: list[str]):
    genes_tfs = selected_genes + [tf for tf in set(dorothea_ab['source']) if tf not in selected_genes]
    mask = torch.zeros(len(genes_tfs), len(genes_tfs), dtype=torch.float)
    idx_to_gt = dict(enumerate(genes_tfs))
    gt_to_idx = {gt: i for i, gt in idx_to_gt.items()}
    gg_pairs = zip(list(dorothea_ab['source']), list(dorothea_ab['target']), list(dorothea_ab['weight']))

    for source, target, w in gg_pairs:
        target_idx = gt_to_idx.get(target, None)
        source_idx = gt_to_idx.get(source, None)
        if target_idx is not None and source_idx is not None:
            mask[target_idx, source_idx] = w

    mask = mask.masked_fill(mask == 0, float('-inf'))
    return {
        'mask': mask,
        'n_genes': len(selected_genes),
        'n_tfs': len(genes_tfs) - len(selected_genes),
    }





def get_trainable_data():
    pseudobulk, dorothea_ab = process_the_data()
    disease_labels = torch.tensor(
        list(
            pseudobulk.obs['disease'].apply(lambda x: 0 if x == 'normal' else 1)
        ), dtype=torch.float
    )

    network_dict = get_network_dict(dorothea_ab, list(pseudobulk.var['feature_name']))
    xpression_count = torch.tensor(pseudobulk.X, dtype=torch.float)
    return xpression_count, disease_labels, network_dict


if __name__ == '__main__':

    data = get_trainable_data()

    # Get the script directory
    script_dir = Path(__file__).parent

    # Create the pickle file path
    pickle_path = script_dir / 'trainable_data.pkl'

    # Save your data
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
