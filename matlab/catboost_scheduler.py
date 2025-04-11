#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import os
import joblib # Можно использовать joblib для кеширования модели
import time

# --- Константы ---
MODEL_DIR = 'ml_model_rb' # Папка для хранения моделей
MODEL_FILE_TEMPLATE = os.path.join(MODEL_DIR, 'catboost_model_RB_{}.cbm') # Модель для DL/UL
CACHE_FILE = os.path.join(MODEL_DIR, 'model_cache_RB.pkl') # Файл для кеширования модели в памяти

# --- Функции ---

def load_model(direction):
    """ Загружает модель или возвращает None, если не найдена """
    model_file = MODEL_FILE_TEMPLATE.format(direction)
    if os.path.exists(model_file):
        try:
            # Пытаемся загрузить из кеша в памяти (если есть)
            if os.path.exists(CACHE_FILE):
                cache_data = joblib.load(CACHE_FILE)
                if cache_data.get('direction') == direction:
                    print(f"RB_Model ({direction}) loaded from memory cache.")
                    return cache_data['model']

            # Загружаем с диска
            print(f"Loading RB Model ({direction}) from file: {model_file}...")
            model = CatBoostClassifier()
            model.load_model(model_file)
            print(f"RB Model ({direction}) loaded successfully.")

            # Сохраняем в кеш
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump({'direction': direction, 'model': model}, CACHE_FILE)
            print(f"RB Model ({direction}) saved to memory cache.")
            return model
        except Exception as e:
            print(f"Error loading RB model {model_file}: {e}")
            # Удаляем поврежденный кеш, если он есть
            if os.path.exists(CACHE_FILE):
                try:
                    os.remove(CACHE_FILE)
                except OSError:
                    pass
            return None
    else:
        print(f"Model file not found: {model_file}")
        return None

def train_model(X_train, y_train, direction, num_ues_hint):
    """ Обучает модель RB предсказывать UE_ID (0..NumUEs) """
    print(f"Training model for {direction}...")
    start_time = time.time()
    
    # Определяем количество классов (0=никто, 1..NumUEs)
    # Используем подсказку NumUEs из MATLAB + 1 (для класса 0)
    num_classes = num_ues_hint + 1
    print(f"Number of classes (based on NumUEs hint + Zero): {num_classes}")
    # Проверяем, есть ли такие классы в y_train
    unique_labels = np.unique(y_train)
    print(f"Unique labels in training data: {unique_labels}")
    if not np.all(np.isin(unique_labels, np.arange(num_classes))):
         print("Warning: Some labels in y_train might be outside the expected range [0, NumUEs].")
         # Можно скорректировать num_classes, если нужно
         # num_classes = int(np.max(y_train)) + 1
         # print(f"Adjusted number of classes to: {num_classes}")
    
    model = CatBoostClassifier(iterations=300, # Можно увеличить
                               learning_rate=0.05,
                               depth=8,
                               l2_leaf_reg=3,
                               loss_function='MultiClass', #'MultiClass' if predicting UE_ID
                               eval_metric='Accuracy', # Или 'AUC'
                               # classes_count=num_classes,
                               custom_metric=['Accuracy', 'AUC'],
                               verbose=100,
                               random_seed=42,
                               early_stopping_rounds=50,
                               # task_type='GPU', # Раскомментировать, если есть GPU и драйверы
                              )

    # Создаем Pool для возможного использования категориальных признаков в будущем
    train_pool = Pool(data=X_train, label=y_train)

    print("Starting model fitting...")
    model.fit(train_pool)#, eval_set=validation_pool)
    end_time = time.time()
    print(f"Training complete. Time elapsed: {end_time - start_time:.2f} seconds")

    # Сохранение модели
    model_file = MODEL_FILE_TEMPLATE.format(direction)
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        model.save_model(model_file)
        print(f"Model saved to {model_file}")
        # Обновляем кеш в памяти
        joblib.dump({'direction': direction, 'model': model}, CACHE_FILE)
        print(f"Model ({direction}) updated in memory cache.")
    except Exception as e:
         print(f"Error saving model {model_file}: {e}")

    return model

def predict_rb_allocation(model, X_predict_rb, num_ues, num_rbs):
    """ Предсказывает UE_ID для каждого RB и формирует матрицу """
    start_time = time.time()
    if model is None:
        print("RB Model is not available. Generating random predictions per RB.")
        # Генерируем случайное решение: для каждого RB выбираем одного UE или никого (0)
        predicted_ue_ids = np.random.randint(0, num_ues + 1, size=num_rbs)
    else:
        if X_predict_rb.shape[0] != num_rbs:
            print(f"ERROR: Input feature rows ({X_predict_rb.shape[0]}) != num_rbs ({num_rbs}). Random prediction.")
            predicted_ue_ids = np.random.randint(0, num_ues + 1, size=num_rbs)
        else:
            print("Predicting UE ID per RB with CatBoost model...")
            predicted_ue_ids = model.predict(X_predict_rb)
            if len(predicted_ue_ids.shape) > 1:
                predicted_ue_ids = predicted_ue_ids.flatten()
            # Ограничиваем предсказания диапазоном [0, NumUEs]
            predicted_ue_ids = np.clip(predicted_ue_ids, 0, num_ues).astype(int)
            print("Prediction complete.")
    # Преобразуем в формат [NumUEs x NumRBs] (0 или 1)
    pred_matrix = np.zeros((num_ues, num_rbs), dtype=int)
    allocated_rbs_mask = np.zeros(num_rbs, dtype=bool) # Отслеживаем, занят ли RB
    # --- Логика распределения: Один UE на RB ---
    # (Модель предсказывает ЛУЧШЕГО UE для RB, поэтому просто ставим 1)
    for rb_idx in range(num_rbs):
        ue_id = predicted_ue_ids[rb_idx]
        if ue_id > 0: # Если предсказан конкретный UE (не 0)
             if ue_id <= num_ues:
                 pred_matrix[ue_id - 1, rb_idx] = 1
             else: # На всякий случай
                 print(f"Warning: Predicted UE ID {ue_id} > NumUEs {num_ues} for RB {rb_idx}. Ignored.")

    # --- Альтернативная логика (если модель предсказывает вероятности) ---
    # Если бы модель предсказывала вероятности (`model.predict_proba`)
    # probs = model.predict_proba(X_predict_rb) # Shape: [NumRBs x (NumUEs + 1)]
    # for rb_idx in range(num_rbs):
    #     best_ue_id_for_rb = np.argmax(probs[rb_idx, 1:]) + 1 # Находим UE с макс. вероятностью (исключая класс 0)
    #     # Дальше нужна логика, чтобы не выделить один RB двум UE (например, жадная)
    #     # ... сложнее ...

    end_time = time.time()
    print(f"Prediction formatting time: {end_time - start_time:.4f} seconds")
    return pred_matrix
    
def main():
    #Получаем аргументы: num_rbs, direction, num_ues
    if len(sys.argv) < 4:
        print("Usage: python catboost_scheduler.py <num_rbs> <direction ('DL' or 'UL')> <num_ues>")
        sys.exit(1)
    try:
        num_rbs = int(sys.argv[1])
        direction = sys.argv[2].upper()
        num_ues = int(sys.argv[3]) # Получаем NumUEs из аргументов
        if direction not in ['DL', 'UL']: raise ValueError("Invalid direction")
        if num_rbs <= 0 or num_ues <= 0: raise ValueError("Num RBs/UEs must be positive")
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        print("Usage: python catboost_scheduler.py <num_rbs> <direction ('DL' or 'UL')> <num_ues>")
        sys.exit(1)

    print(f"\n--- Running CatBoost RB Scheduler ---")
    print(f"Num RBs: {num_rbs}, Direction: {direction}, Num UEs: {num_ues}")
    overall_start_time = time.time()

    # --- Загрузка данных состояния ---
    state_file = 'current_state.csv'
    try:
        print(f"Loading state data from {state_file}...")
        # Замеряем время чтения
        read_start = time.time()
        # Используем numpy.loadtxt для скорости, если формат простой
        # current_state_df = pd.read_csv(state_file, header=None)
        # X_predict_rb = current_state_df.values
        X_predict_rb = np.loadtxt(state_file, delimiter=',')
        read_end = time.time()
        print(f"State data loading time: {read_end - read_start:.4f} seconds")

        if X_predict_rb.ndim == 1: # Если только один RB, loadtxt вернет 1D массив
             X_predict_rb = X_predict_rb.reshape(1, -1) # Преобразуем в 2D

        if X_predict_rb.shape[0] != num_rbs:
            print(f"ERROR: Rows in {state_file} ({X_predict_rb.shape[0]}) != num_rbs ({num_rbs})")
            np.savetxt('predictions.csv', np.zeros((num_ues, num_rbs), dtype=int), delimiter=',', fmt='%d')
            sys.exit(1)
        num_features_per_rb = X_predict_rb.shape[1]
        print(f"Loaded current state: {num_rbs} RBs, {num_features_per_rb} Features per RB")
    except FileNotFoundError:
        print(f"Error: {state_file} not found.")
        np.savetxt('predictions.csv', np.zeros((num_ues, num_rbs), dtype=int), delimiter=',', fmt='%d')
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {state_file}: {e}")
        np.savetxt('predictions.csv', np.zeros((num_ues, num_rbs), dtype=int), delimiter=',', fmt='%d')
        sys.exit(1)

    # --- Загрузка или Обучение Модели ---
    model = load_model(direction)

    if model is None:
        training_data_file = f'training_data_rb_{direction}.csv' # Отдельные файлы для DL/UL
        if os.path.exists(training_data_file):
            print(f"Training data found ({training_data_file}). Training new RB model...")
            try:
                # Замеряем время чтения данных обучения
                read_train_start = time.time()
                # train_df = pd.read_csv(training_data_file, header=None)
                train_data = np.loadtxt(training_data_file, delimiter=',')
                read_train_end = time.time()
                print(f"Training data loading time: {read_train_end - read_train_start:.4f} seconds")

                if train_data.ndim == 1: # Если всего одна строка данных
                    train_data = train_data.reshape(1, -1)

                num_cols = train_data.shape[1]
                if num_cols <= 1: raise ValueError("Training data needs at least one feature and one target column.")

                X_train = train_data[:, :num_cols-1]
                y_train = train_data[:, num_cols-1].astype(int) # Последняя колонка - ID UE (0..NumUEs)

                # Проверяем согласованность NumUEs
                max_label = int(np.max(y_train))
                if max_label > num_ues:
                    print(f"Warning: Max label in training data ({max_label}) > provided NumUEs ({num_ues}). Adjusting NumUEs.")
                    # num_ues = max_label # Не можем менять num_ues здесь, он нужен для матрицы выхода

                model = train_model(X_train, y_train, direction, num_ues) # Передаем подсказку num_ues
            except Exception as e:
                print(f"Error processing training data '{training_data_file}': {e}")
                model = None
        else:
            print(f"Training data '{training_data_file}' not found. Using random predictions.")

    # --- Предсказание ---
    predictions_matrix = predict_rb_allocation(model, X_predict_rb, num_ues, num_rbs)

    # --- Сохранение результатов ---
    save_start = time.time()
    try:
        np.savetxt('predictions.csv', predictions_matrix, delimiter=',', fmt='%d')
        save_end = time.time()
        print(f"Predictions saved to predictions.csv ({predictions_matrix.shape}). Save time: {save_end - save_start:.4f} seconds")
    except Exception as e:
         print(f"Error saving predictions.csv: {e}")

    overall_end_time = time.time()
    print(f"--- CatBoost RB Scheduler Finished. Total time: {overall_end_time - overall_start_time:.3f} seconds ---")

if __name__ == '__main__':
    main()