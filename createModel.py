from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import EarlyStopping

def create_and_train_model(X_train_scaled, Y_train, X_val_scaled, Y_val, n):
    # Model Architecture
    model = Sequential()
    model.add(Dense(110, input_dim=2, activation='relu'))
    model.add(Dense(110, activation='relu'))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(12))  # 12 outputs corresponding to target values

    model.summary()
    # Use a learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate, decay_steps=100, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', 'mae', 'mse'])

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Model Training
    history = model.fit(X_train_scaled, Y_train, epochs=n, batch_size=35,
                        validation_split= 0.3, verbose=0)

    return model, history