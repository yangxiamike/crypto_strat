from sklearn.preprocessing import MinMaxScaler

config = {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": -1,
    "num_leaves": 31,
    "test_size": 0.2,
    "scaler": MinMaxScaler(),
    "early_stopping_rounds": 200,

    "lgbm_params": {
        "boosting_type": "gbdt",
        "feature_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": 0,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "random_state": 42,
        "min_child_samples": 20,
        "min_split_gain": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
}
