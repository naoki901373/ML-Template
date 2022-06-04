from sklearn.model_selection import KFold
fold = 5

def feat_eng(df):
    skf = KFold(n_splits = fold, shuffle=True, random_state=72)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
        dataframe.loc[val_idx, 'fold'] = int(fold)
    return  df