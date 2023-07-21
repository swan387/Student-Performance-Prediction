import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch


@torch.no_grad()
def test_and_val(y_score, y, mode='test', epoch=0):
    y_score = y_score.cpu().numpy()
    y = y.cpu().numpy()
    y_pred = np.where(y_score >= .5, 1, 0)
    res = {
        f'{mode}_epoch': epoch,
        f'{mode}_pos_ratio': np.sum(y) / len(y),
        f'{mode}_auc': roc_auc_score(y, y_pred),
        f'{mode}_f1': f1_score(y, y_pred),
        f'{mode}_macro_f1': f1_score(y, y_pred, average='macro'),
        f'{mode}_micro_f1': f1_score(y, y_pred, average='micro')
    }
    return res


def save_as_df(dict_list, path: str, show=True, append=False) -> None:
    import os
    if not os.path.exists(path):
        append = False
        print('file not exists, set append to False.')

    if append:
        df = pd.read_pickle(path)
        df = pd.concat([df, pd.DataFrame.from_dict(dict_list).round(3)], axis=0)
    else:
        df = pd.DataFrame.from_dict(dict_list).round(3)  # create dataframe
    df.to_pickle(path)  # save file
    if show:
        print(df)
        print('Result saved.')


def load_df(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)
