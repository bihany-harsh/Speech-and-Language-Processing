import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from sklearn.metrics.pairwise import cosine_similarity

class Word2Vec:
    def __init__(self, corpus, vector_dim=100, window_size=5, num_ns=5):
        self.corpus = corpus
        self.vector_dim = vector_dim
        self.window_size = window_size
        self.num_ns = num_ns
        self.tokenizer = Tokenizer()
        self.vocabulary_size = None
        self.model = None

    def preprocess_corpus(self):
        self.tokenizer.fit_on_texts(self.corpus)
        self.vocabulary_size = len(self.tokenizer.word_index) + 1

    def generate_training_data(self):
        sequences = self.tokenizer.texts_to_sequences(self.corpus)
        targets, contexts, labels = [], [], []
        for sequence in sequences:
            sg = skipgrams(sequence, self.vocabulary_size, window_size=self.window_size, negative_samples=self.num_ns)
            for pair, label in zip(sg[0], sg[1]):
                targets.append(pair[0])
                contexts.append(pair[1])
                labels.append(label)
        return tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

    def build_model(self):
        target_word = tf.keras.layers.Input((1,), dtype='int32')
        context_word = tf.keras.layers.Input((1,), dtype='int32')

        embedding = tf.keras.layers.Embedding(self.vocabulary_size, self.vector_dim)

        target_embedding = embedding(target_word)
        context_embedding = embedding(context_word)

        # Reshape the embedding outputs
        target_embedding = tf.keras.layers.Reshape((self.vector_dim, 1))(target_embedding)
        context_embedding = tf.keras.layers.Reshape((self.vector_dim, 1))(context_embedding)

        dot_product = tf.keras.layers.Dot(axes=(2, 2))([target_embedding, context_embedding])
        dot_product = tf.keras.layers.Reshape((1,))(dot_product)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

        self.model = tf.keras.Model(inputs=[target_word, context_word], outputs=output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')




    def train_model(self, dataset, epochs=5):
        self.model.fit(dataset, epochs=epochs)

    def get_word_embeddings(self):
        weights = self.model.get_weights()[0]
        embeddings = {w: weights[idx] for w, idx in self.tokenizer.word_index.items()}
        return embeddings

    def get_word_embeddings(self):
        weights = self.model.get_weights()[0]
        self.embeddings = {w: weights[idx] for w, idx in self.tokenizer.word_index.items()}

    def most_similar_words(self, word, n=10):
        if self.embeddings is None:
            raise Exception("You need to call get_word_embeddings() first.")

        word_vector = self.embeddings.get(word, None)
        if word_vector is None:
            raise Exception(f"Unknown word: {word}")

        similarity_scores = {}
        for w, w_vec in self.embeddings.items():
            if w != word:
                similarity_scores[w] = cosine_similarity([word_vector], [w_vec])

        most_similar_words = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
        return most_similar_words[:n]
