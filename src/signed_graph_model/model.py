import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class CombineLayer(nn.Module):
    """
    Combine the embedding from GNN and NLP semantic embedding
    """

    def __init__(self, args):
        super(CombineLayer, self).__init__()
        self.lin_qus = nn.Linear(2 * args.emb_size, args.emb_size, bias=False)
        self.lin_usr = nn.Linear(args.emb_size, args.emb_size, bias=False)
        self.lin_out = nn.Linear(2 * args.emb_size, 1)  # predict the sign
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=.1)

    def forward(self, z, edge_index, z0):
        """predict the sign of edge given GNN output embedding, edge index and question semantic embeddings
        z: GNN output embeddings
        z0: raw question embeddings
        """
        qid = torch.max(edge_index[0:2], dim=0).values  # get question indices
        qid -= qid.min()
        emb1, emb2 = z[edge_index[0]], z[edge_index[1]]  # user and question embedding from GNN
        x0 = self.lin_qus(torch.cat([emb2, z0[qid]], dim=-1))
        return self.lin_out(torch.cat([emb1, x0], dim=-1)).flatten()


class GATCL(torch.nn.Module):
    """Use contrastive learning to get an embedding for each node"""

    def __init__(self, args):
        super(GATCL, self).__init__()
        self.args = args
        self.emb_size = args.emb_size
        self.emb, self.norm_emb = None, None  # final embeddings

        self.layer_ab_pos = torch.nn.ModuleList([GATConv(self.emb_size, self.emb_size) for _ in range(args.num_layers)])
        self.layer_ab_neg = torch.nn.ModuleList([GATConv(self.emb_size, self.emb_size) for _ in range(args.num_layers)])
        self.linear_combine = torch.nn.Linear(4 * self.emb_size, self.emb_size, bias=False)
        self.activation = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.link_mlp = LinkMLP(args)  # make prediction

    def forward(self, x, edge_index_g1_pos, edge_index_g2_pos, edge_index_g1_neg, edge_index_g2_neg):
        emb_g1_pos = x
        emb_g2_pos = x
        emb_g1_neg = x
        emb_g2_neg = x

        # positive edges
        for layer in self.layer_ab_pos:
            # graph 1
            emb_g1_pos = layer(emb_g1_pos, edge_index_g1_pos)
            emb_g1_pos = self.activation(emb_g1_pos)
            # graph 2
            emb_g2_pos = layer(emb_g2_pos, edge_index_g2_pos)
            emb_g2_pos = self.activation(emb_g2_pos)

        # negative edges
        for layer in self.layer_ab_neg:
            # graph 1
            emb_g1_neg = layer(emb_g1_neg, edge_index_g1_neg)
            emb_g1_neg = self.activation(emb_g1_neg)
            # graph 2
            emb_g2_neg = layer(emb_g2_neg, edge_index_g2_neg)
            emb_g2_neg = self.activation(emb_g2_neg)

        # dropout
        emb_g1_pos, emb_g2_pos = self.dropout(emb_g1_pos), self.dropout(emb_g2_pos)
        emb_g1_neg, emb_g2_neg = self.dropout(emb_g1_neg), self.dropout(emb_g2_neg)

        # final embeddings
        self.emb = self.linear_combine(torch.cat([emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg], dim=-1))
        self.norm_emb = self.norm_emb = F.normalize(self.emb, p=2, dim=1)

        return emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg

    def predict_edges(self, emb, uid, qid):
        """Predict the sign of edges given embeddings and user id, question id"""
        usr_emb = emb[uid]
        qus_emb = emb[qid]
        return self.link_mlp(usr_emb, qus_emb)

    def compute_label_loss(self, y_score, y_label):
        pos_weight = torch.tensor([(y_label == 0).sum().item() / (y_label == 1).sum().item()] * y_label.shape[0]).to(
            y_score.device)
        return F.binary_cross_entropy_with_logits(y_score, y_label, pos_weight=pos_weight)

    def compute_contrastive_loss(self, emb_g1_pos, emb_g2_pos, emb_g1_neg, emb_g2_neg):
        nodes_num, feature_size = emb_g1_pos.shape

        norm_emb_g1_pos = F.normalize(emb_g1_pos, p=2, dim=1)
        norm_emb_g2_pos = F.normalize(emb_g2_pos, p=2, dim=1)
        norm_emb_g1_neg = F.normalize(emb_g1_neg, p=2, dim=1)
        norm_emb_g2_neg = F.normalize(emb_g2_neg, p=2, dim=1)

        def inter_contrastive(embs_attr, embs_stru):
            pos = torch.exp(torch.div(
                torch.bmm(embs_attr.view(nodes_num, 1, feature_size), embs_stru.view(nodes_num, feature_size, 1)),
                self.args.tau))

            def generate_neg_score(emb_1, emb_2):
                neg_similarity = torch.mm(emb_1.view(nodes_num, feature_size), emb_2.transpose(0, 1)).fill_diagonal_(0)
                return torch.sum(torch.exp(torch.div(neg_similarity, self.args.tau)), dim=1)

            neg = generate_neg_score(embs_attr, embs_stru)

            return torch.mean(- (torch.log(torch.div(pos, neg))))

        def intra_contrastive(self_embs, embs_attr_pos, embs_attr_neg, embs_stru_pos, embs_stru_neg):
            pos_score_1 = torch.exp(torch.div(
                torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_attr_pos.view(nodes_num, feature_size, 1)),
                self.args.tau))
            pos_score_2 = torch.exp(torch.div(
                torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_stru_pos.view(nodes_num, feature_size, 1)),
                self.args.tau))
            pos = pos_score_1 + pos_score_2

            def generate_neg_score(pos_embs, neg_embs_1, neg_embs_2):
                neg_score_1 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size),
                                        neg_embs_1.view(nodes_num, feature_size, 1))
                neg_score_2 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size),
                                        neg_embs_2.view(nodes_num, feature_size, 1))
                return torch.exp(torch.div(neg_score_1, self.args.tau)) + torch.exp(
                    torch.div(neg_score_2, self.args.tau))

            neg = generate_neg_score(self_embs, embs_attr_neg, embs_stru_neg)
            return torch.mean(- torch.log(torch.div(pos, neg)))

        inter_pos = inter_contrastive(norm_emb_g1_pos, norm_emb_g2_pos)
        inter_neg = inter_contrastive(norm_emb_g1_neg, norm_emb_g2_neg)

        intra = intra_contrastive(self.norm_emb, norm_emb_g1_pos, norm_emb_g1_neg,
                                  norm_emb_g2_pos, norm_emb_g2_neg)
        return (1 - self.args.alpha) * (inter_pos + inter_neg) + self.args.alpha * intra


class LinkMLP(torch.nn.Module):
    """Predict the sign of edges using the learned embeddings"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.linear_predictor_layers == 0:  # use dot product
            pass
        elif args.linear_predictor_layers == 1:
            self.predictor = torch.nn.Linear(2 * args.emb_size, 1)
        elif args.linear_predictor_layers == 2:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(2 * args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, 1))
        elif args.linear_predictor_layers == 3:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(2 * args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, 1))
        elif args.linear_predictor_layers == 4:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(2 * args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, args.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(args.emb_size, 1))
        else:
            raise NotImplementedError("Invalid layer number.")

    def forward(self, v_user: torch.Tensor, v_qust: torch.Tensor):
        if self.args.linear_predictor_layers == 0:  # dot product
            return v_user.mul(v_qust).sum(dim=-1)
        x = torch.cat([v_user, v_qust], dim=-1)  # concat the user and question embeddings
        return self.predictor(x).flatten()
