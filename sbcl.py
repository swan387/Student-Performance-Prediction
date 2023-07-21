if __name__ == '__main__':
    import os.path as osp
    import copy
    import torch
    from tqdm import tqdm
    from torch_geometric import seed_everything
    from src.signed_graph_model.model import GATCL
    from utils.results import test_and_val
    from utils.load_data import (
        split_train_val_test,
        graph_augmentation,
        results_mean_std
    )
    import joblib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding dimension for each node.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN (implemented by pyg) layers.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout parameter.')
    parser.add_argument('--linear_predictor_layers', type=int, default=1, choices=range(5),
                        help='Number of MLP layers (0-4) to make prediction from learned embeddings.')
    parser.add_argument('--mask_ratio', type=float, default=0.1, help='Random mask ratio')
    parser.add_argument('--beta', type=float, default=5e-4, help='Control contribution of loss contrastive.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Control the contribution of inter and intra loss.')
    parser.add_argument('--tau', type=float, default=0.05, help='Temperature parameter.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=301, help='Number of epochs.')
    parser.add_argument('--dataset', type=str, default='biology', help='The dataset to be used.')
    parser.add_argument('--rounds', type=int, default=1, help='Repeating the training and evaluation process.')
    args = parser.parse_args()
    print(args)

    # init settings
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    data_info = joblib.load(osp.join('datasets', args.dataset, 'data_info.pkl'))  # load data info

    # model
    seed_everything(args.seed)
    model = GATCL(args).to(device)
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    x = torch.randn(size=(data_info['num_user'] + data_info['num_ques'], args.emb_size)).to(device)


    def run(round_i: int):
        model.load_state_dict(model_state_dict)  # reset parameters

        edge_index = torch.load(osp.join('datasets', args.dataset, 'edge_index.pt'))
        (g_train, g_val, g_test), (y_train, y_val, y_test) = \
            split_train_val_test(edge_index, ratios=[.85, .05, .1], device=device)  # split train-val-test
        edge_index_g1_pos, edge_index_g1_neg, edge_index_g2_pos, edge_index_g2_neg = \
            graph_augmentation(g_train, ratio=args.mask_ratio)  # graph augmentation

        # train the model
        best_res = {'val_auc': 0, 'val_f1': 0}

        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg = model(x, edge_index_g1_pos, edge_index_g2_pos,
                                                                   edge_index_g1_neg, edge_index_g2_neg)
            # contrastive loss
            loss_contrastive = model.compute_contrastive_loss(emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg)
            y_score = model.predict_edges(model.emb, g_train[0], g_train[1])
            loss = args.beta * loss_contrastive + model.compute_label_loss(y_score, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                _ = model(x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg)
                y_score_val = model.predict_edges(model.emb, g_val[0], g_val[1])
            val_res = test_and_val(y_score_val, y_val, mode='val', epoch=epoch)

            if val_res['val_auc'] + val_res['val_f1'] > best_res['val_auc'] + best_res['val_f1'] and epoch >= 100:
                best_res.update(val_res)
                y_score_test = model.predict_edges(model.emb, g_test[0], g_test[1])
                best_res.update(test_and_val(y_score_test, y_test, mode='test', epoch=epoch))

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]
    results_mean_std(results)
