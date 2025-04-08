# AdvancedFocalTuner
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, roc_auc_score, f1_score
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import Hyperband
from kerastuner import Objective  # ← 指定目标方向

class AdvancedFocalTuner:
    def __init__(self,
                 max_epochs=50,
                 factor=3,
                 executions_per_trial=1,
                 use_smote=True,
                 patience=5,
                 scaler=None):
        """
        max_epochs, factor, executions_per_trial: Hyperband 参数
        use_smote: 是否对训练集做 SMOTE
        patience: EarlyStopping 容忍轮数
        scaler: sklearn 标准化器
        """
        self.max_epochs = max_epochs
        self.factor = factor
        self.executions_per_trial = executions_per_trial
        self.use_smote = use_smote
        self.patience = patience
        self.scaler = scaler or StandardScaler()
        self.tuner = None
        self.best_model = None
        self.best_hp = None
        self.best_threshold = None

    def _focal_loss(self, alpha, gamma):
        def loss_fn(y_true, y_pred):
            y_true = K.cast(y_true, 'float32')
            eps = K.epsilon()
            y_pred = K.clip(y_pred, eps, 1. - eps)
            ce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
            weight = alpha * K.pow(1 - y_pred, gamma) * y_true \
                   + (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
            return K.mean(weight * ce, axis=-1)
        return loss_fn

    def build_model(self, hp):
        # 网络结构超参
        units1   = hp.Int('units1', 64, 512, step=64)
        dropout1 = hp.Float('dropout1', 0.2, 0.6, step=0.1)
        units2   = hp.Int('units2', 32, 256, step=32)
        dropout2 = hp.Float('dropout2', 0.2, 0.5, step=0.1)
        # focal loss 超参
        alpha    = hp.Float('alpha', 0.1, 0.9, step=0.1)
        gamma    = hp.Int('gamma', 1, 5, step=1)
        # 学习率
        lr       = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

        model = Sequential([
            Dense(units1, activation='relu', input_dim=self.input_dim),
            Dropout(dropout1),
            Dense(units2, activation='relu'),
            Dropout(dropout2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=self._focal_loss(alpha, gamma),
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def tune(self, X_train, y_train, X_val, y_val):
        # 1. 转 float32 & 标准化
        y_train = y_train.astype('float32')
        y_val   = y_val.astype('float32')
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)

        # 2. SMOTE（可选）
        if self.use_smote:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # 3. class_weight
        neg, pos = np.bincount(y_train.astype(int))
        cw = {0: 1.0, 1: neg/pos}

        # 4. 准备 Hyperband，指定 objective 为最大化 val_auc
        self.input_dim = X_train.shape[1]
        self.tuner = Hyperband(
            hypermodel=self.build_model,
            objective=Objective('val_auc', direction='max'),
            max_epochs=self.max_epochs,
            factor=self.factor,
            executions_per_trial=self.executions_per_trial,
            directory='hyperband_dir',
            project_name='adv_focal'
        )
        stop_early = EarlyStopping(monitor='val_loss', patience=self.patience)
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            callbacks=[stop_early],
            class_weight=cw,
            verbose=1
        )

        # 5. 获取最优
        self.best_hp    = self.tuner.get_best_hyperparameters(1)[0]
        self.best_model = self.tuner.get_best_models(1)[0]
        print("✔️ Hyperband 最佳超参：")
        for p in ['units1','dropout1','units2','dropout2','alpha','gamma','learning_rate']:
            print(f"  - {p}: {self.best_hp.get(p)}")

        # 6. 用 F1 在验证集选阈值
        self.best_threshold = self._select_threshold_f1(X_val, y_val)
        print("✔️ 验证集 F1 最佳阈值：", self.best_threshold)
        return self.best_model, self.best_hp, self.best_threshold

    def _select_threshold_f1(self, X, y):
        y_prob = self.best_model.predict(X).ravel()
        prec, rec, ths = precision_recall_curve(y, y_prob)
        f1s = 2 * prec * rec / (prec + rec + K.epsilon())
        ix = np.nanargmax(f1s)
        return ths[ix] if ix < len(ths) else 0.5

    def evaluate(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        y_prob = self.best_model.predict(X_test).ravel()
        y_pred = (y_prob >= self.best_threshold).astype(int)

        cm      = confusion_matrix(y_test, y_pred)
        report  = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1      = f1_score(y_test, y_pred)

        print("混淆矩阵：\n", cm)
        print("\n分类报告：\n", report)
        print(f"ROC AUC: {roc_auc:.4f}, F1-score: {f1:.4f}")
        return {'cm': cm, 'report': report, 'roc_auc': roc_auc, 'f1': f1}


if __name__ == "__main__":
    # 1. 数据预处理（同前）
    data = pd.read_csv('merged_data.csv', encoding='utf-8')
    data['行业大类'] = data['行业代码'].str.extract(r'([A-Za-z]+)')
    data = pd.concat([data, pd.get_dummies(data['行业大类'], prefix='Industry')], axis=1)
    data.drop(columns=['行业代码','行业大类','行业名称','证券简称'], inplace=True)
    data.interpolate(method='linear', inplace=True)
    data.dropna(inplace=True)

    X = data.drop(columns=['股票代码','公司简称','会计年度','是否违规'])
    y = data['是否违规'].astype(int)

    # 2. 划分（加 stratify 保持分布）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

    # 3. 调参 + 评估
    tuner = AdvancedFocalTuner(
        max_epochs=50,
        factor=3,
        executions_per_trial=1,
        use_smote=True,
        patience=5
    )
    model, best_hp, best_thresh = tuner.tune(X_train, y_train, X_val, y_val)
    results = tuner.evaluate(X_te, y_te)


# output
'''
Reloading Tuner from hyperband_dir/adv_focal/tuner0.json
✔️ Hyperband 最佳超参：
  - units1: 448
  - dropout1: 0.5
  - units2: 128
  - dropout2: 0.4
  - alpha: 0.5
  - gamma: 4
  - learning_rate: 0.0001
254/254 [==============================] - 0s 802us/step
✔️ 验证集 F1 最佳阈值： 0.47142556
318/318 [==============================] - 0s 802us/step
混淆矩阵：
 [[8338 1367]
 [ 308  142]]

分类报告：
               precision    recall  f1-score   support

           0       0.96      0.86      0.91      9705
           1       0.09      0.32      0.14       450

    accuracy                           0.84     10155
   macro avg       0.53      0.59      0.53     10155
weighted avg       0.93      0.84      0.87     10155

ROC AUC: 0.6642, F1-score: 0.1450
'''