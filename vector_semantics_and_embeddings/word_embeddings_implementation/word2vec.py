import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Word2Vec:
    def __init__(self, corpus, vector_dims=50, window_size=5, num_ns=2):
        self.corpus = corpus
        self.vector_dims = vector_dims
        self.window_size = window_size
        self.num_ns = num_ns
        self.tokenizer = Tokenizer()
        self.vocab_size = None
        self.model = None

    def preprocess_corpus(self):
        self.tokenizer.fit_on_texts(self.corpus)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def generate_training_data(self):
        sequences = self.tokenizer.texts_to_sequences(self.corpus)

        pairs = []
        labels = []

        for sequence in sequences:
            pairs_i, labels_i = skipgrams(
                sequence,
                vocabulary_size=self.vocab_size,
                window_size=self.window_size,
                negative_samples=self.num_ns,
            )
            pairs.extend(pairs_i)
            labels.extend(labels_i)

        pairs = np.array(pairs)
        labels = np.array(labels)
        return pairs, labels

    def build_model(self):
        input_target = tf.keras.layers.Input((1,))
        input_context = tf.keras.layers.Input((1,))

        embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.vector_dims, input_length=1, name="embedding"
        )

        target = embedding(input_target)
        target = tf.keras.layers.Reshape((self.vector_dims, 1))(target)

        context = embedding(input_context)
        context = tf.keras.layers.Reshape((self.vector_dims, 1))(context)

        dot_product = tf.keras.layers.Dot(axes=1)([target, context])
        dot_product = tf.keras.layers.Reshape((1,))(dot_product)

        output = tf.keras.layers.Dense(1, activation="sigmoid")(dot_product)

        self.model = tf.keras.Model(inputs=[input_target, input_context], outputs=output)
        self.model.compile(loss="binary_crossentropy", optimizer="adam")

    def train(self, epochs=100, batch_size=1024):
        pairs, labels = self.generate_training_data()
        history = self.model.fit(
            x=[pairs[:, 0], pairs[:, 1]],
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
        )
        return history
    
    def get_embedding(self, word):
        word_index = self.tokenizer.word_index[word]
        return self.model.get_layer("embedding").get_weights()[0][word_index]

    def get_similar_words(self, word, top_n=5):
        word_embedding = self.get_embedding(word)
        all_embeddings = self.model.get_layer("embedding").get_weights()[0]

        all_embeddings_normalized = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        word_embedding_normalized = word_embedding / np.linalg.norm(word_embedding)

        similarity_scores = cosine_similarity(
            all_embeddings_normalized, word_embedding_normalized.reshape(1, -1)
        ).flatten()
        similarity_scores[self.tokenizer.word_index[word]] = -2

        sorted_indexes = np.argsort(similarity_scores)[::-1][:top_n]
        similar_words = [(self.tokenizer.index_word[i], similarity_scores[i]) for i in sorted_indexes]
        return similar_words

