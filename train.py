from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

def build_lstm_model(input_shape, vocab_size):
    """ Define the neural network architecture """
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model(model, X, y):
    """ Train the model and save optimal weights """
    checkpoint = ModelCheckpoint(
        "best_weights.h5",
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    print("Training is starting... (This may take a while depending on hardware)")
    model.fit(X, y, epochs=30, batch_size=64, callbacks=[checkpoint])
    return model