import random
from trainer.trainer import TrainGraphEncoder, TrainBaselineEncoder, compare_models
import pickle
from pathlib import Path


if __name__ == "__main__":
    # load the processed data
    print("\n" + "=" * 80)
    print("LOADING THE PROCESSED DATA")
    print("=" * 80)

    script_dir = Path(__file__).parent  # Main_folder/tasks/
    main_folder = script_dir.parent  # Main_folder/
    pickle_path = main_folder / 'Data' / 'trainable_data.pkl'

    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    x, y, network_dict = data

    # Fix common parameters
    lr = 1e-3
    num_epochs = int(5e3)
    batch_size = 64
    # embedding dimension
    emb_dim = 32

    # split data
    idxs = list(range(x.shape[0]))
    random.shuffle(idxs)
    n = len(idxs)
    n_t, n_test = int(0.8*n), int(0.9*n)
    train_idxs, val_idxs, test_idxs = idxs[: n_t], idxs[n_t: n_test], idxs[n_test:]
    train_data = (x[train_idxs], y[train_idxs], x[val_idxs], y[val_idxs])
    test_data = (x[test_idxs], y[test_idxs])


    # Train baseline model
    architect_params = {
        'x_dim': x.shape[1],
        'hidden_dim': 5*emb_dim,
        'n_layers': 4,
        'emb_dim': emb_dim,
    }

    base_trainer = TrainBaselineEncoder(architect_params, lr, num_epochs, batch_size, train_data)
    base_trainer.run_epochs()
    base_model = base_trainer.model.eval()


    # Train graph model
    architect_params = {
        'x_dim': x.shape[1],
        'n_genes': network_dict['n_genes'],
        'n_tfs': network_dict['n_tfs'],
        'n_heads': 2,
        'head_size': int(emb_dim/4),
        'n_att_blocks': 1,
        'mask': network_dict['mask'],
    }
    mask = network_dict['mask']

    graph_trainer = TrainGraphEncoder(architect_params, lr, num_epochs, batch_size, train_data)
    graph_trainer.run_epochs()
    graph_model = graph_trainer.model.eval()

    # compare the models
    compare_models(
        graph_model=graph_model,
        baseline_model=base_model,
        data=test_data,
        mask=mask,
    )





