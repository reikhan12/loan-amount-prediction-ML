# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import time

# Veri Yükleme
data = pd.read_csv('/Users/reyhanqurbanova/Downloads/financial_risk_assessment.csv')

# Veri Temizleme ve Ön İşleme
selected_columns = [
    'Age', 'Income', 'Credit Score', 'Loan Amount', 'Years at Current Job',
    'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults', 'Risk Rating'
]
data = data[selected_columns].copy()

for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        data[col] = data[col].fillna(data[col].median())
if data['Risk Rating'].isnull().sum() > 0:
    data['Risk Rating'] = data['Risk Rating'].fillna(data['Risk Rating'].mode()[0])

label_encoder = LabelEncoder()
data['Risk Rating Encoded'] = label_encoder.fit_transform(data['Risk Rating'])

# Regresyon için Veri Seti
X = data.drop(columns=['Loan Amount', 'Risk Rating', 'Risk Rating Encoded'])
y = data['Loan Amount']

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Sonuçları İçin Sözlükler
results = {}
training_times = {}

# Learning Rate Scheduler ve EarlyStopping Callback'ler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 1. Geleneksel ML Modelleri
traditional_models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor()
}

for name, model in traditional_models.items():
    print(f"\n Model: {name}")
    start_time = time.time()
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    end_time = time.time()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"MSE": mse, "MAE": mae}
    training_times[name] = end_time - start_time
    print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, Training Time: {training_times[name]:.2f} seconds")

# 2. Deep Learning Modelleri

def create_deep_ann(layers_config=[128, 64, 32, 64, 32], dropout_ratio=0.3):
    model = Sequential()
    for i, neurons in enumerate(layers_config):
        if i == 0:
            model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
        else:
            model.add(Dense(neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_ratio))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model

def create_lstm(layers_config=[128, 64], dropout_ratio=0.3):
    model = Sequential()
    for i, neurons in enumerate(layers_config):
        if i == 0:
            model.add(LSTM(neurons, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        else:
            model.add(LSTM(neurons, activation='tanh', return_sequences=(i != len(layers_config) - 1)))
        model.add(Dropout(dropout_ratio))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model

def create_gru(layers_config=[128, 64], dropout_ratio=0.3):
    model = Sequential()
    for i, neurons in enumerate(layers_config):
        if i == 0:
            model.add(GRU(neurons, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        else:
            model.add(GRU(neurons, activation='tanh', return_sequences=(i != len(layers_config) - 1)))
        model.add(Dropout(dropout_ratio))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model

# LSTM ve GRU için veri hazırlığı
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



dropOutList = [0.1, 0.2, 0.3, 0.4, 0.5]
annconfiglist = [
    [128, 64, 32, 64, 32],
    [256, 128, 64, 32, 16],
    [512, 256, 128, 64, 32],
    [64, 32, 16, 32, 16],
    [128, 128, 64, 32, 16]
]
lstmconfiglist = [
    [128, 64],
    [64, 32],
    [256, 128],
    [128, 32],
    [64, 64]
]
gruconfiglist = [
    [128, 64],
    [64, 32],
    [256, 128],
    [128, 32],
    [64, 64]
]

resultsList = []


for k in range(len(dropOutList)):

    for i in range(len(annconfiglist)):

        nn_models = {
            "Dynamic ANN": create_deep_ann(annconfiglist[i], dropOutList[k]),
            "LSTM": create_lstm(lstmconfiglist[i], dropOutList[k]),
            "GRU": create_gru(gruconfiglist[i], dropOutList[k])
        }

        nn_history = {}
        row=[]
        for name, model in nn_models.items():
            print(f"\nEğitilen Model: {name}")
            start_time = time.time()
            if name in ["LSTM", "GRU"]:
                history = model.fit(
                    X_train_lstm, y_train, 
                    validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
                    epochs=100, batch_size=128, verbose=1, callbacks=[lr_scheduler, early_stopping]
                )
                y_pred = model.predict(X_test_lstm).flatten()
            else:
                history = model.fit(
                    X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=100, batch_size=128, verbose=1, callbacks=[lr_scheduler, early_stopping]
                )
                y_pred = model.predict(X_test).flatten()
            end_time = time.time()

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {"MSE": mse, "MAE": mae}
            training_times[name] = end_time - start_time
            nn_history[name] = history
            print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, Training Time: {training_times[name]:.2f} seconds")
            row.append(mse)

        # 3. Sonuçların Görselleştirilmesi
        # Adjusting the plot layout for readability
        # Adjusting the plot layout for readability
        
        
        
        # Adjust the plot layout and figure size
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Reduce figure height for smaller subplots

        # MSE Comparison
        axs[0].bar(results.keys(), [v["MSE"] for v in results.values()], color='skyblue')
        axs[0].set_title("Model Comparison - MSE", fontsize=14)  # Reduce title font size
        axs[0].set_ylabel("Mean Squared Error (MSE)", fontsize=12)
        axs[0].tick_params(axis='x', labelsize=10)
        axs[0].tick_params(axis='y', labelsize=10)
        axs[0].set_xticklabels(results.keys(), rotation=30, ha='right')  # Rotate labels at 30 degrees

        # MAE Comparison
        axs[1].bar(results.keys(), [v["MAE"] for v in results.values()], color='lightgreen')
        axs[1].set_title("Model Comparison - MAE", fontsize=14)  # Reduce title font size
        axs[1].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
        axs[1].tick_params(axis='x', labelsize=10)
        axs[1].tick_params(axis='y', labelsize=10)
        axs[1].set_xticklabels(results.keys(), rotation=30, ha='right')  # Rotate labels at 30 degrees

        # Training Time Comparison
        axs[2].bar(training_times.keys(), training_times.values(), color='salmon')
        axs[2].set_title("Model Comparison - Training Time", fontsize=14)  # Reduce title font size
        axs[2].set_ylabel("Training Time (seconds)", fontsize=12)
        axs[2].tick_params(axis='x', labelsize=10)
        axs[2].tick_params(axis='y', labelsize=10)
        axs[2].set_xticklabels(training_times.keys(), rotation=30, ha='right')  # Rotate labels at 30 degrees

        # Increase space between subplots and adjust layout
        plt.subplots_adjust(hspace=1.2)  # Add more vertical space between subplots
        plt.tight_layout()
        plt.show()
        plt.close()  # Close the plot after displaying

        
        resultsList.append(row)
        # Neural Network Validation Loss Grafiği
        plt.figure(figsize=(10, 6))
        for name, history in nn_history.items():
            plt.plot(history.history['val_loss'], label=f"{name} Validation Loss")

        plt.title("Validation Loss per Epoch for Neural Networks")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig('res'+f'{i}'+'.png')

        plt.show()
        plt.close()


        resultsList.append(row)  # Sonuçları kaydettiğiniz kısım.

print("\n25 Run Sonuçları (Konfigürasyon ve Modellerin Performans Değerleri):")
for i, result in enumerate(resultsList, start=1):
    # Validation loss değerlerini çek
    dynamic_ann_val_loss = nn_history["Dynamic ANN"].history["val_loss"][-1]
    lstm_val_loss = nn_history["LSTM"].history["val_loss"][-1]
    gru_val_loss = nn_history["GRU"].history["val_loss"][-1]
    
    print(
        f"Run {i:02d}: "
        f"ANN - MSE = {result[0]:.4f}, MAE = {results['Dynamic ANN']['MAE']:.4f}, "
        f"Training Time = {training_times['Dynamic ANN']:.2f} s, Val Loss = {dynamic_ann_val_loss:.4f} | "
        f"LSTM - MSE = {result[1]:.4f}, MAE = {results['LSTM']['MAE']:.4f}, "
        f"Training Time = {training_times['LSTM']:.2f} s, Val Loss = {lstm_val_loss:.4f} | "
        f"GRU - MSE = {result[2]:.4f}, MAE = {results['GRU']['MAE']:.4f}, "
        f"Training Time = {training_times['GRU']:.2f} s, Val Loss = {gru_val_loss:.4f}"
    )


print(resultsList)


best_dropout = None  # To store the best dropout rate
best_val_loss = float('inf')  # To store the lowest validation loss

for k in range(len(dropOutList)):
    for i in range(len(annconfiglist)):
        nn_models = {
            "Dynamic ANN": create_deep_ann(annconfiglist[i], dropOutList[k]),
            "LSTM": create_lstm(lstmconfiglist[i], dropOutList[k]),
            "GRU": create_gru(gruconfiglist[i], dropOutList[k])
        }

        nn_history = {}
        for name, model in nn_models.items():
            print(f"\nTraining Model: {name}")
            if name in ["LSTM", "GRU"]:
                history = model.fit(
                    X_train_lstm, y_train, 
                    validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
                    epochs=100, batch_size=128, verbose=1, callbacks=[lr_scheduler, early_stopping]
                )
            else:
                history = model.fit(
                    X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=100, batch_size=128, verbose=1, callbacks=[lr_scheduler, early_stopping]
                )
            
            # Check validation loss (val_loss)
            current_val_loss = history.history['val_loss'][-1]  # Last epoch's val_loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_dropout = dropOutList[k]  # Record the best dropout rate
                best_model = name  # Record the best model
        
        print(f"Run {k}-{i} Best Model: {best_model}, Dropout: {best_dropout}, Val Loss: {best_val_loss:.4f}")

# Print the best dropout rate
print(f"\nBest dropout rate: {best_dropout}, Lowest val_loss: {best_val_loss:.4f}")
