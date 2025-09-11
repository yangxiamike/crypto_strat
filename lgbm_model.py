import os
import math
from scipy.fftpack import shift
import yaml
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from add_features import add_technical_indicators

from configs import lgbm_configs

class LGBMClassifierModel:



    def __init__(self, config_path='configs/LGBM_Config.yaml'):
        self.config = lgbm_configs.config
        self.ensure_directories()
        self.scaler_x = self.config['scaler']

    def ensure_directories(self):
        """Create necessary directories for plots and models if they don't exist."""
        for dir_name in ['plots', 'models', 'logs']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    def load_data(self, exchange, ticker, timeframe, start, end, crypto_type='spot'):
        """
        Load historical OHLCV data from CSV file.

        Args:
            exchange (str): Exchange name.
            ticker (str): Ticker symbol, {BASE}/{QUOTE} format
            timeframe (str): Timeframe for the data.
            start (str): Start date for data filtering.
            end (str): End date for data filtering.
        Returns:
            pd.DataFrame: DataFrame containing the OHLCV data.
        """
        file_path = f"data/{crypto_type}/{exchange}/{timeframe}/{ticker.replace('/', '_')}.csv"
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
        df.set_index('datetime', inplace=True)
        return df

    def get_features(self, df):

        # 调用特征库构建特征
        df = add_technical_indicators(df)
        return df
    
    def get_target(self, df, period=1):
        # 二分类目标变量
        df['Target'] = (df['close'].shift(-period) > df['close']).astype(int)
        return df
    
    def split_data(self, df):
        df = df.dropna()

        feature_cols = [col for col in df.columns if col != 'Target']
        target_col = 'Target'

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], shuffle=False)

        X_train = self.scaler_x.fit_transform(X_train)
        X_test = self.scaler_x.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, X_test, y_test):

        params = self.config['lgbm_params'].copy()

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        early_stopping_callback = lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds'], verbose=True)

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping_callback]
        )

        return model
    
    def evaluate(self, model, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC:", roc_auc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def grid(self, X_train, y_train, X_test, y_test):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training target.
            X_test (np.array): Testing features.
            y_test (np.array): Testing target.

        Returns:
            tuple: GridSearchCV object and best parameters.
        """
        param_grid = self.config['param_grid']

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        early_stopping_callback = lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds'], verbose=True)

        grid_search = GridSearchCV(
            estimator=lgb.LGBMRegressor(**self.config['lgbm_params']),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=self.config['cv_folds'],
            verbose=2
        )

        grid_search.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[early_stopping_callback])

        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

        return grid_search, grid_search.best_params_

    def yhat(self, ticker, model, X_test, y_test):
        """
        Make predictions and calculate RMSE.

        Args:
            ticker (str): Stock ticker symbol.
            model (lgb.Booster): Trained LightGBM model.
            X_test (np.array): Testing features.
            y_test (np.array): Testing target.

        Returns:
            float: Root Mean Squared Error (RMSE).
        """
        yhat = model.predict(X_test)
        y_test = self.scaler_y.inverse_transform(y_test)
        yhat = self.scaler_y.inverse_transform(yhat.reshape(-1, 1))

        rmse = math.sqrt(mean_squared_error(y_test, yhat))

        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Price')
        plt.plot(yhat, label='Predicted Price')
        plt.title(f'{ticker} Price Prediction - LGBM Model')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'plots/LGBM_{ticker}_prediction.png')
        plt.close()

        return rmse

    def save_model(self, model, ticker):
        """
        Save the trained model and scalers.

        Args:
            model (lgb.Booster): Trained LightGBM model.
            ticker (str): Stock ticker symbol.
        """
        joblib.dump(model, f'models/{ticker}_lgbm_model.joblib')
        joblib.dump(self.scaler_x, f'models/{ticker}_lgbm_scaler_x.joblib')
        joblib.dump(self.scaler_y, f'models/{ticker}_lgbm_scaler_y.joblib')

    def load_model(self, ticker):
        """
        Load a saved model and scalers.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            lgb.Booster: Loaded LightGBM model.
        """
        model = joblib.load(f'models/{ticker}_lgbm_model.joblib')
        self.scaler_x = joblib.load(f'models/{ticker}_lgbm_scaler_x.joblib')
        self.scaler_y = joblib.load(f'models/{ticker}_lgbm_scaler_y.joblib')
        return model
    


    def run(self, ticker, start_date, end_date):
        """
        Run the entire modeling process: data preparation, training, and evaluation.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for data download.
            end_date (str): End date for data download.

        Returns:
            tuple: Trained model and RMSE.
        """
        X_train, X_test, y_train, y_test = self.yfdown(ticker, start_date, end_date)
        grid_search, best_params = self.grid(X_train, y_train, X_test, y_test)
        model = self.model(X_train, y_train, X_test, y_test, best_params)
        rmse = self.yhat(ticker, model, X_test, y_test)
        print(f'RMSE: {rmse}')
        self.save_model(model, ticker)
        return model, rmse

    def predict_new_data(self, model, ticker, start_date, end_date):
        """
        Make predictions on new data.

        Args:
            model (lgb.Booster): Trained LightGBM model.
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for new data.
            end_date (str): End date for new data.

        Returns:
            tuple: RMSE, dates, actual prices, and predicted prices.
        """
        # 下载新数据
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.dropna()

        # 添加技术指标
        df = self.get_features(df)
        df = df.dropna()

        # 构建特征集
        feature_cols = self.config.get('feature_columns', [
            'Prev_Close', 'Prev_SMA_20', 'Prev_EMA_12', 'Prev_RSI', 'Day_of_Week', 'Volume', 'Open', 'High', 'Low'
        ])
        X = df[feature_cols]
        X_scaled = self.scaler_x.transform(X)

        # 预测
        y_pred = model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        # 真实值
        y_true = df['Close'].values.reshape(-1, 1)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, y_true, label='Actual Price')
        plt.plot(df.index, y_pred, label='Predicted Price')
        plt.title(f'{ticker} Price Prediction - LGBM Model (New Data)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'plots/LGBM_{ticker}_prediction_new_data.png')
        plt.close()

        return rmse, df.index, y_true, y_pred


if __name__ == "__main__":
    import os
    os.chdir('加密货币')
    lgbm_model = LGBMClassifierModel()
    data = lgbm_model.load_data('binance', 'BTC/USDT', '15m', '2023-01-01', '2024-06-22')
    data = lgbm_model.get_features(data)
    data= lgbm_model.get_target(data, period=3)
    X_train, X_test, y_train, y_test = lgbm_model.split_data(data)
    model = lgbm_model.train(X_train, y_train, X_test, y_test)
    lgbm_model.evaluate(model, X_train, y_train)
    lgbm_model.evaluate(model, X_test, y_test)

