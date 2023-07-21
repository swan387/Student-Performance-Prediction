import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_qus_emb_from_pt(path: str):
    """
    load question id and their embeddings, given file path
    """
    qus_emb_raw_ = torch.load(path)
    qus_emb_df_ = pd.concat([pd.Series(qus_emb_raw_['question_id'], name='question_id'),
                             pd.Series(qus_emb_raw_['embedding'], name='question_emb')],
                            axis=1)
    return qus_emb_df_


def _load_qus_emb_from_csv(path: str):
    """
    load question id and their embeddings, given file path
    """
    qus_emb_df_ = pd.read_csv(path).rename(columns={'ID': 'question_id', 'embedding': 'question_emb'})
    qus_emb_df_['question_emb'] = qus_emb_df_['question_emb'].transform(lambda x: np.array([x]))
    return qus_emb_df_


def _load_ans_qus_old(folder_path: str):
    """
    load answer and question from UOA
    """
    ans_path_, qus_path_ = osp.join(folder_path, 'Answers_CourseX.xlsx'), osp.join(folder_path,
                                                                                   'Questions_CourseX.xlsx')
    # load answer
    ans_raw_ = pd.read_excel(ans_path_)
    ans_raw_ = ans_raw_.rename(columns={'UserID': 'user_id', 'QuestionID': 'question_id', 'Answer': 'answer'})
    ans_raw_ = ans_raw_[['user_id', 'question_id', 'answer']]

    # load question
    qus_raw_ = pd.read_excel(qus_path_)
    qus_raw_ = qus_raw_.rename(columns={'QuestionID': 'question_id', 'Answer': 'answer'})
    qus_raw_ = qus_raw_[['question_id', 'answer']]

    return ans_raw_, qus_raw_


def _load_txt(path: str) -> pd.DataFrame:
    """
    read the txt file into a dataframe
    """

    def get_title() -> list[str]:
        with open(path) as f:
            f.readline()
            t = f.readline().split("|")
            return list(filter(None, list(map(str.strip, t))))  # strip the title and remove empty element

    title_ = get_title()
    n_cols_ = len(title_)

    df_ = pd.read_csv(path, sep="|", usecols=range(1, n_cols_ + 1), header=0, names=title_, dtype=object,
                      encoding="unicode-escape")
    df_.drop([0, 1, df_.shape[0] - 1], axis=0, inplace=True)  # drop the line seperator (e.g. +--------+)
    df_.reset_index(drop=True, inplace=True)
    return df_


def _load_ans_qus_new(folder_path: str, course_id: int):
    """
    load answer and question from other sources
    """
    ans_path_, qus_path_ = osp.join(folder_path, 'All_Answers.txt'), osp.join(folder_path, 'All_Questions.txt')
    # read records
    type_dict_ans_ = {
        'user': np.int64,
        'question_id': np.int64,
        'answer': str,
        'course_id': np.int64
    }
    ans_raw_ = _load_txt(ans_path_)[['user', 'question_id', 'answer', 'course_id']].astype(type_dict_ans_)
    ans_raw_['answer'] = ans_raw_['answer'].str.strip()
    ans_raw_ = ans_raw_[ans_raw_['course_id'] == course_id] \
        .drop(columns='course_id') \
        .rename(columns={'user': 'user_id'}) \
        .reset_index(drop=True)

    # read questions
    type_dict_qus_ = {
        'id': np.int64,
        'course_id': np.int64,
        'total_responses': np.int64,
        'deleted': np.int64,
        'answer': str
    }
    qus_raw_ = _load_txt(qus_path_)
    qus_raw_ = qus_raw_[~qus_raw_['id'].isna()]  # filter the NA rows
    qus_raw_ = qus_raw_[['id', 'course_id', 'total_responses', 'deleted', 'answer']].astype(type_dict_qus_)
    qus_raw_['answer'] = qus_raw_['answer'].str.strip()
    # subset questions
    filter_mask_ = (qus_raw_['course_id'] == course_id) & (qus_raw_['total_responses'] > 0) & (qus_raw_['deleted'] == 0)
    qus_raw_ = qus_raw_[filter_mask_][['id', 'answer']].reset_index(drop=True).rename(columns={'id': 'question_id'})

    return ans_raw_, qus_raw_


def _create_edge_index_qus_emb_df_filtered(ans_raw_, qus_raw_, qus_emb_df_):
    from sklearn.preprocessing import LabelEncoder

    merged_ = ans_raw_.merge(qus_raw_, on='question_id').merge(qus_emb_df_, on='question_id')
    merged_['sign'] = (merged_['answer_x'] == merged_['answer_y']).astype(int) * 2 - 1
    merged_ = merged_.drop(columns=['answer_x', 'answer_y'])

    usr_encoder_, qus_encoder_ = LabelEncoder(), LabelEncoder()

    # process edge index
    edge_index_ = merged_.copy()[['user_id', 'question_id', 'sign']]
    edge_index_['user_id'] = usr_encoder_.fit_transform(edge_index_['user_id'])
    edge_index_['question_id'] = qus_encoder_.fit_transform(edge_index_['question_id'])

    # save data information
    num_user_ = edge_index_['user_id'].nunique()
    num_ques_ = edge_index_['question_id'].nunique()
    data_info_ = {
        'num_nodes': num_user_ + num_ques_,
        'num_user': num_user_,
        'num_ques': num_ques_,
        'num_links': len(edge_index_),
        'pos_links': (edge_index_['sign'] > 0).sum(),
        'neg_links': (edge_index_['sign'] < 0).sum()
    }
    edge_index_['question_id'] += data_info_['num_user']
    edge_index_ = torch.tensor(edge_index_.values)

    # process filtered question embedding base
    qus_emb_df_filtered_ = merged_.copy()[['question_id', 'question_emb']].drop_duplicates(subset=['question_id'])
    qus_emb_df_filtered_['question_id'] = qus_encoder_.transform(qus_emb_df_filtered_['question_id'])
    qus_emb_df_filtered_ = qus_emb_df_filtered_.sort_values(by='question_id')

    return edge_index_, qus_emb_df_filtered_, data_info_, usr_encoder_, qus_encoder_


def _create_qus_emb(qus_emb_df_filtered_, repeat_count=None):
    """
    Create embedding tensor for questions from the dataframe
    """
    qus_emb_ = torch.tensor(np.array(qus_emb_df_filtered_['question_emb'].tolist()))
    if repeat_count is not None:
        qus_emb_ = qus_emb_.repeat(1, repeat_count)
    return qus_emb_.float()


def _create_usr_emb_from_qus_emb(edge_index_: torch.Tensor, qus_emb_tensor_, data_info_, args_, add_noise=False):
    from sklearn.preprocessing import StandardScaler

    user_emb_ = torch.zeros(data_info_['num_user'], args_.emb_size)
    num_user_ = data_info_['num_user']

    # for each user (manuel message passing)
    for uid, qid, sign in edge_index_:
        user_emb_[uid] += sign * qus_emb_tensor_[qid - num_user_]

    # fill zeros using normal distribution
    n_zeros_ = (torch.sum(user_emb_, dim=1) == 0).sum()
    user_emb_[torch.sum(user_emb_, dim=1) == 0] = torch.randn(n_zeros_, args.emb_size)

    X_ = torch.tensor(StandardScaler().fit_transform(user_emb_.numpy()))

    if add_noise:
        pass
    return X_


def split_train_val_test(edge_index_, ratios: list, device: str):
    """
    Split an edge index, return the train, validation and test sets, and their labels
    edge_index_: [n_edges, 3]
    """
    trn_id, val_id, tst_id = torch.utils.data.random_split(range(edge_index_.size(0)), ratios)
    trn_egi, val_egi, tst_egi = edge_index_[trn_id], edge_index_[val_id], edge_index_[tst_id]

    g_trn, g_val, g_tst = trn_egi.t().to(device), val_egi.t().to(device), tst_egi.t().to(device)
    y_trn, y_val, y_tst = (g_trn[2] == 1).float(), (g_val[2] == 1).float(), (g_tst[2] == 1).float()

    return (g_trn, g_val, g_tst), (y_trn, y_val, y_tst)


def graph_augmentation(g, ratio):
    """
    Augment a graph by flipping sign of edges, return 2 views
    g: [3, n_edges]
    """
    mask1 = torch.ones(g.size(1), dtype=torch.bool)
    mask2 = torch.ones(g.size(1), dtype=torch.bool)
    mask1[torch.randperm(mask1.size(0))[:int(ratio * mask1.size(0))]] = 0
    mask2[torch.randperm(mask2.size(0))[:int(ratio * mask2.size(0))]] = 0
    g1, g2 = g.clone(), g.clone()
    g1[2, ~mask1] *= -1
    g2[2, ~mask2] *= -1
    egi_g1_pos = g1[0:2, g1[2] > 0]  # graph 1
    egi_g1_neg = g1[0:2, g1[2] < 0]
    egi_g2_pos = g2[0:2, g2[2] > 0]  # graph 2
    egi_g2_neg = g2[0:2, g2[2] < 0]

    return egi_g1_pos, egi_g1_neg, egi_g2_pos, egi_g2_neg


def graph_augmentation_graph(g, ratio):
    """
    Augment a graph by flipping sign of edges, return 2 views
    g: [3, n_edges]
    """
    mask1 = torch.ones(g.size(1), dtype=torch.bool)
    mask2 = torch.ones(g.size(1), dtype=torch.bool)
    mask1[torch.randperm(mask1.size(0))[:int(ratio * mask1.size(0))]] = 0
    mask2[torch.randperm(mask2.size(0))[:int(ratio * mask2.size(0))]] = 0
    g1, g2 = g.clone(), g.clone()
    g1[2, ~mask1] *= -1
    g2[2, ~mask2] *= -1
    return g1, g2


def split_pos_neg_edges(g):
    g_pos = g[0:2, g[2] > 0]
    g_neg = g[0:2, g[2] < 0]
    return g_pos, g_neg


def update_res_dict_from_batch(res_dict, res_dict_batch, mode='val'):
    res_dict[f'{mode}_auc'] += res_dict_batch[f'{mode}_auc']
    res_dict[f'{mode}_f1'] += res_dict_batch[f'{mode}_f1']
    res_dict[f'{mode}_macro_f1'] += res_dict_batch[f'{mode}_macro_f1']
    res_dict[f'{mode}_micro_f1'] += res_dict_batch[f'{mode}_micro_f1']


def res_dict_avg(res_dict, num_batch, mode='val'):
    res_dict[f'{mode}_auc'] /= num_batch
    res_dict[f'{mode}_f1'] /= num_batch
    res_dict[f'{mode}_macro_f1'] /= num_batch
    res_dict[f'{mode}_micro_f1'] /= num_batch


def results_mean_std(res_list):
    """
    show results as a data frame, showing mean and std
    """
    cols_vis = ['test_f1', 'test_micro_f1', 'test_macro_f1', 'test_auc']
    res_df = pd.DataFrame.from_dict(res_list)[cols_vis]
    print(res_df)
    m = res_df[cols_vis].mean().round(3)
    s = res_df[cols_vis].std().round(3)
    print(m.astype(str) + r'$\pm$' + s.astype(str))


if __name__ == '__main__':
    import os.path as osp
    from dotmap import DotMap

    args = DotMap(
        dataset='law',
        course_id=20102,
        new_data=False,
        method='glove_300d',  # glove_1d_old, glove_1d_new, glove_300d, glove_1d_cls
        emb_size=64,
        seed=42
    )

    # load data
    if args.new_data:
        FOLD_PATH = osp.join('..', 'datasets', 'Sydney_Cardiff_PW_Data', args.dataset.capitalize())
        QUES_EMB_PATH = osp.join('..', 'embeddings', args.dataset + str(args.course_id), f'{args.method}.pt')
        ans_raw, qus_raw = _load_ans_qus_new(FOLD_PATH, args.course_id)
    else:
        FOLD_PATH = osp.join('..', 'datasets', 'PeerWiseData', args.dataset.capitalize())
        QUES_EMB_PATH = osp.join('..', 'embeddings', args.dataset, f'{args.method}.pt')
        ans_raw, qus_raw = _load_ans_qus_old(FOLD_PATH)

    # load question embedding
    if args.method == 'glove_1d_new':
        QUES_EMB_PATH = osp.join('..', 'embeddings', args.dataset, f'{args.method}.csv')
        qus_emb_df = _load_qus_emb_from_csv(QUES_EMB_PATH)
    else:
        qus_emb_df = _load_qus_emb_from_pt(QUES_EMB_PATH)

    # create edge index
    edge_index, qus_emb_df_filtered, data_info, usr_encoder, qus_encoder = \
        _create_edge_index_qus_emb_df_filtered(ans_raw, qus_raw, qus_emb_df)

    # save edge_index, data_info
    if args.new_data:
        torch.save(edge_index, osp.join('..', 'datasets', args.dataset + str(args.course_id), 'edge_index.pt'))
        joblib.dump(data_info, osp.join('..', 'datasets', args.dataset + str(args.course_id), 'data_info.pkl'))
    else:
        torch.save(edge_index, osp.join('..', 'datasets', args.dataset, 'edge_index.pt'))
        joblib.dump(data_info, osp.join('..', 'datasets', args.dataset, 'data_info.pkl'))

    # save question embedding
    if '300d' in args.method:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        qus_emb_tensor = _create_qus_emb(qus_emb_df_filtered)
        # NOTE: PCA (300d -> 64d)
        X = StandardScaler().fit_transform(qus_emb_tensor.numpy())
        pca = PCA(n_components=args.emb_size, whiten=True)
        X_pca = pca.fit_transform(X)
        qus_emb_tensor = torch.tensor(X_pca)
    else:
        qus_emb_tensor = _create_qus_emb(qus_emb_df_filtered, repeat_count=args.emb_size)

    if args.new_data:
        torch.save(qus_emb_tensor, osp.join('..', 'datasets', args.dataset + str(args.course_id), f'{args.method}.pt'))
    else:
        torch.save(qus_emb_tensor, osp.join('..', 'datasets', args.dataset, f'{args.method}.pt'))
