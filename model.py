import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam

def build_model(max_words=10000, embedding_dim=16, max_len=200):

    model = tf.keras.Sequential([
        L.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        L.GlobalAveragePooling1D(),
        L.Dense(16, activation='relu'),
        L.Dropout(0.5),
        L.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, train_padded, y_train, test_padded, y_test, epochs=10):
    history = model.fit( train_padded, y_train, epochs=epochs, validation_data=(test_padded, y_test), verbose=1)
    return history
