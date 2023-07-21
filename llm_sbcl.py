if __name__ == '__main__':
    import os.path as osp
    import copy
    import torch
    from tqdm import tqdm
    from torch_geometric import seed_everything
    from src.signed_graph_model.model import GATCL, CombineLayer
    from utils.results import test_and_val
    from utils.load_data import (
        split_train_val_test,
        graph_augmentation,
        results_mean_std,
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
    parser.add_argument('--dataset', type=str, default='sydney19351', help='The dataset to be used.')
    parser.add_argument('--rounds', type=int, default=1, help='Repeating the training and evaluation process.')
    args = parser.parse_args()
    print(args)

    # init settings
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    data_info = joblib.load(osp.join('datasets', args.dataset, 'data_info.pkl'))  # load data info
    z0 = torch.load(osp.join('datasets', args.dataset, 'glove_300d.pt')).to(device)

    # model
    seed_everything(args.seed)
    model, combl = GATCL(args).to(device), CombineLayer(args).to(device)
    model_params, combl_params = copy.deepcopy(model.state_dict()), copy.deepcopy(combl.state_dict())
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': 5e-4},
                                  {'params': combl.parameters()}], lr=args.lr)
    x = torch.randn(size=(data_info['num_user'] + data_info['num_ques'], args.emb_size)).to(device)


    def run(round_i: int):
        model.load_state_dict(model_params)  # reset parameters
        combl.load_state_dict(combl_params)

        edge_index = torch.load(osp.join('datasets', args.dataset, 'edge_index.pt'))
        (g_train, g_val, g_test), (y_train, y_val, y_test) = split_train_val_test(edge_index, ratios=[.85, .05, .1],
                                                                                  device=device)  # split train-val-test
        edge_index_g1_pos, edge_index_g1_neg, edge_index_g2_pos, edge_index_g2_neg = graph_augmentation(g_train,
                                                                                                        ratio=args.mask_ratio)  # graph augmentation

        # train the model
        best_res = {'val_auc': 0, 'val_f1': 0}

        for epoch in tqdm(range(args.epochs)):
            model.train()
            combl.train()
            optimizer.zero_grad()
            emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg = model(x, edge_index_g1_pos, edge_index_g2_pos,
                                                                   edge_index_g1_neg, edge_index_g2_neg)
            # contrastive loss
            loss_contrastive = model.compute_contrastive_loss(emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg)
            y_score = combl(model.emb, g_train, z0)
            loss = args.beta * loss_contrastive + model.compute_label_loss(y_score, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            combl.eval()
            with torch.no_grad():
                _ = model(x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg)
                y_score_val = combl(model.emb, g_val, z0)
            val_res = test_and_val(y_score_val, y_val, mode='val', epoch=epoch)

            if val_res['val_auc'] + val_res['val_f1'] > best_res['val_auc'] + best_res['val_f1'] and epoch >= 100:
                best_res.update(val_res)
                y_score_test = combl(model.emb, g_test, z0)
                best_res.update(test_and_val(y_score_test, y_test, mode='test', epoch=epoch))

        print(f'Round {round_i} done.')
        return best_res


    results = [run(i) for i in range(args.rounds)]
    results_mean_std(results)
