import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score

class TextCNNClassifier:
    def __init__(self, max_sequence_length, vocab_size, embedding_dim, num_classes):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.max_sequence_length = max_sequence_length
        self.model = self.build_model(embedding_dim, num_classes)

    def build_model(self, embedding_dim, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim, input_length=self.max_sequence_length),
            tf.keras.layers.Conv1D(256, 10, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs, batch_size):
        self.tokenizer.fit_on_texts(X_train)
        x_train_padded = self.preprocess_text(X_train)
        y_train = np.array(y_train)
        self.model.fit(x_train_padded, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def preprocess_text(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences

    def evaluate(self, x_test, y_test):
        x_test_padded = self.preprocess_text(x_test)
        y_pred = self.model.predict(x_test_padded)
        y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        return precision, recall, f1
