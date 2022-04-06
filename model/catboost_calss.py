from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier


model_list = []
mae_list = []

# fold5
kf = KFold(n_splits = 5, shuffle = True, random_state = 70)

# modeling and training
for fold, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
    acc = 0
    print(f'--------fold:{fold+1}--------')
    fold+=1
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    params = {
        'loss_function' : 'Logloss',
        'task_type' : 'GPU', 
        'grow_policy' : 'SymmetricTree',
        'learning_rate': 0.1,
        'l2_leaf_reg' : 0.2,
        'random_state': 0
     }
                  
    model = CatBoostClassifier(**params)
    # Training the model
    
    model.fit(tr_x,
              tr_y,
              eval_set=[(va_x, va_y)])
    
    val_pred = model.predict(va_x)
    print(f' ROC: {roc_auc_score(va_y, val_pred)}')