from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor


model_list = []
mae_list = []

# fold5
kf = KFold(n_splits = 5, shuffle = True, random_state = 70)

# modeling and training
for fold, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
    print(f'--------fold:{fold+1}--------')
    fold+=1
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    params = {'logging_level': 'Silent',
              'depth': 12,
              'eval_metric': 'MAE',
              'loss_function': 'MAE',
              'n_estimators': 800,
              'task_type': 'GPU'
        
     }
                  
    model = CatBoostRegressor(**params)
    # Training the model
    
    model.fit(tr_x,
              tr_y,
              eval_set=[(va_x, va_y)])
    
    val_pred = model.predict(va_x)
    
    print(f' MAE: {mean_absolute_error(va_y, val_pred)}')
    