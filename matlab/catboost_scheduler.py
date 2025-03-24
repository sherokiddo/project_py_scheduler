#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os

def main():
    # Определяем число ресурсных блоков (RB) из аргумента (по умолчанию 10)
    if len(sys.argv) < 2:
        rb_count = 10
    else:
        rb_count = int(sys.argv[1])
    
    # Если обучающие данные отсутствуют, выдаем случайные предсказания
    if not os.path.isfile('training_data.csv'):
        current_state = pd.read_csv('current_state.csv', header=None)
        num_samples = current_state.shape[0]
        predictions = np.random.randint(0, 2, (num_samples, rb_count))
        np.savetxt('predictions.csv', predictions, delimiter=',', fmt='%d')
        return
    
    # Читаем обучающие данные; общее число столбцов = NumFeatures + rb_count.
    train_data = pd.read_csv('training_data.csv', header=None)
    total_cols = train_data.shape[1]
    num_features = total_cols - rb_count  # число признаков (например: 5 для CQI_DL, CQI_UL, RSRP, TrafficDL, TrafficUL)
    X_train = train_data.iloc[:, :num_features].values
    y_train_all = train_data.iloc[:, num_features:].values
    
    models = []
    for rb in range(rb_count):
        y_train = y_train_all[:, rb]
        # Используем GPU для обучения модели (параметр task_type='GPU')
        model = CatBoostClassifier(iterations=100, verbose=False, random_seed=42, task_type='GPU')
        model.fit(X_train, y_train)
        models.append(model)
    
    current_state = pd.read_csv('current_state.csv', header=None)
    current_state = current_state.iloc[:, :num_features]
    X_state = current_state.values
    num_samples = X_state.shape[0]
    predictions = np.zeros((num_samples, rb_count), dtype=int)
    for rb in range(rb_count):
        preds = models[rb].predict(X_state)
        predictions[:, rb] = preds.astype(int)
    np.savetxt('predictions.csv', predictions, delimiter=',', fmt='%d')

if __name__ == '__main__':
    main()