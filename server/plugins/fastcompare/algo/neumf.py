import os.path
import numpy as np
import pandas as pd
from abc import ABC
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, utils
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType

class NeuMFWrapper(AlgorithmBase, ABC):

    def __init__(self, loader, GMF_embedding_size, MLP_embedding_size, negatives_count, diversity_factor, epochs, batch_size, pretrained_model_name, **kwargs):

        utils.set_random_seed(42)

        assert 0 < GMF_embedding_size
        self._GMF_embedding_size = GMF_embedding_size

        assert 0 < MLP_embedding_size
        self._MLP_embedding_size = MLP_embedding_size

        assert 0 <= diversity_factor and diversity_factor <= 1
        self._diversity_factor = diversity_factor

        assert 0 < negatives_count
        self._negatives_count = negatives_count

        assert 0 < epochs
        self._epochs = epochs

        assert 0 < batch_size
        self._batch_size = batch_size

        self._pretrained_model_name = None if pretrained_model_name == "-" else pretrained_model_name

        self._ratings_df: pd.DataFrame = loader.ratings_df.copy()
        self._ratings_df.drop_duplicates(subset=[ "user", "item" ])

        self._interaction_matrix = np.where((self._ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values) > 0, 1, 0)

        self._new_user = self._ratings_df["user"].max() + 1

        self._unique_users = np.concatenate([ self._ratings_df["user"].unique(), [ self._new_user ] ])
        self._unique_items = self._ratings_df["item"].unique()

        self._model = None

    def fit(self):

        if self._pretrained_model_name:

            return

        ratings_df = self._ratings_df

        # Create positive interactions

        users_pos = ratings_df["user"].to_numpy()
        items_pos = ratings_df["item"].to_numpy()
        interactions_pos = np.ones(ratings_df.shape[0])

        # Create negative interactions

        interaction_pairs_neg = []

        for user in self._unique_users:

            pos = ratings_df[ratings_df["user"] == user]["item"].to_numpy()

            if pos.shape[0] == 0: continue

            neg = np.setdiff1d(self._unique_items, pos)

            selected_neg = []

            for _ in range(self._negatives_count * pos.shape[0]):

                item_neg = np.random.choice(neg)
                
                while item_neg in selected_neg:

                    item_neg = np.random.choice(neg)

                interaction_pairs_neg.append((user, item_neg))

        users_neg = np.array([ u for u, _ in interaction_pairs_neg ])
        items_neg = np.array([ i for _, i in interaction_pairs_neg ])
        interactions_neg = np.zeros(len(interaction_pairs_neg))

        # Merge positive and negative interactionons

        self._users_train = np.concatenate([ users_pos, users_neg ])
        self._items_train = np.concatenate([ items_pos, items_neg ])
        self._interactions_train = np.concatenate([ interactions_pos, interactions_neg ])

        # Fit

        self._model = NeuMF(self._unique_users, self._unique_items, self._GMF_embedding_size, self._MLP_embedding_size, [ 32, 16, 8 ])
        self._model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy")
        self._model.fit([ self._users_train, self._items_train ], self._interactions_train, batch_size=self._batch_size, epochs=self._epochs)

    def predict(self, selected_items, filter_out_items, k):

        # Fine-tune model

        new_user = np.full((len(selected_items),), self._new_user)
        new_user_items_pos = np.array(selected_items)
        new_user_interactions_pos = np.ones(len(selected_items))

        for _ in range(8):

            fine_tune_users = np.random.choice(np.arange(self._unique_users.shape[0]), size=64, replace=False)
            fine_tune_items = np.random.choice(np.arange(self._unique_items.shape[0]), size=64, replace=False)
            fine_tune_interactions = self._interaction_matrix[fine_tune_users, fine_tune_items]

            fine_tune_users = np.concatenate([ new_user, fine_tune_users ], axis=None)
            fine_tune_items = np.concatenate([ new_user_items_pos,  fine_tune_items ], axis=None)
            fine_tune_interactions = np.concatenate([ new_user_interactions_pos, fine_tune_interactions ], axis=None)

            self._model.fit([ fine_tune_users, fine_tune_items ], fine_tune_interactions, batch_size=8, epochs=1)

        # Predict

        items_to_predict = np.setdiff1d(self._unique_items, filter_out_items)
        items_to_predict = np.setdiff1d(items_to_predict, selected_items)
        users_to_predict = np.full((len(items_to_predict),), self._new_user)

        probs = (self._model.predict([ users_to_predict, items_to_predict ])).flatten()

        candidates_idxs = np.argsort(probs)[-k*5:]

        return self._get_diversified_top_k(items_to_predict[candidates_idxs], probs[candidates_idxs], self._diversity_factor, k)

    def save(self, instance_cache_path, class_cache_path):

        if not self._pretrained_model_name:

            self._model.save(instance_cache_path + ".keras")

    def load(self, instance_cache_path, class_cache_path):

        if self._pretrained_model_name:

            dirname = os.path.dirname(instance_cache_path)
            path = os.path.join(dirname, self._pretrained_model_name + ".keras")

        else:

            path = instance_cache_path + ".keras"

        self._model = tf.keras.models.load_model(path, custom_objects={ "NeuMF": NeuMF })

    def _get_diversified_top_k(self, items, probs, diversity_factor, k):

        item_lookup_layer = self._model.get_layer(name="item_lookup")
        item_embedding_layer = self._model.get_layer(name="MLP_item_embedding")

        embedding = lambda i: item_embedding_layer(item_lookup_layer(i))
        cosine_similarity = lambda u, v: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

        selected_items = []

        best_item = items[np.argmax(probs)]
        selected_items.append(best_item)

        while len(selected_items) < k:

            best_item = None
            best_score = -np.inf

            for item, prob in zip(items, probs):

                if item not in selected_items:

                    max_similarity = max([ cosine_similarity(embedding(item), embedding(selected_item)) for selected_item in selected_items ])

                    mmr = (1 - diversity_factor) * prob - diversity_factor * max_similarity

                    if mmr > best_score:

                        best_item = item
                        best_score = mmr

            selected_items.append(best_item)

        return selected_items

    @classmethod
    def name(cls):

        return "NeuMF"

    @classmethod
    def parameters(cls):

        return [
            Parameter(
                "GMF_embedding_size",
                ParameterType.INT,
                20
            ),
            Parameter(
                "MLP_embedding_size",
                ParameterType.INT,
                32
            ),
            Parameter(
                "negatives_count",
                ParameterType.INT,
                5,
                "Number of negative samples per positive sample."
            ),
            Parameter(
                "epochs",
                ParameterType.INT,
                10
            ),
            Parameter(
                "batch_size",
                ParameterType.INT,
                1024
            ),
            Parameter(
                "diversity_factor",
                ParameterType.FLOAT,
                0.0001,
                "Diversification strength: allowed values are from interval [0, 1]."
            ),
            Parameter(
                "pretrained_model_name",
                ParameterType.STRING,
                "-",
                "Use \"-\" if you want to train new model; otherwise, specify name of existing model to skip fitting. This is useful for experimenting with various diversity factors."
            )
        ]

class NeuMF(Model):

    def __init__(
        self,
        unique_users,
        unique_items,
        GMF_embedding_size,
        MLP_embedding_size,
        hidden_layers_units,
        **kwargs
    ):

        self.unique_users = unique_users
        self.unique_items = unique_items
        self.GMF_embedding_size = GMF_embedding_size
        self.MLP_embedding_size = MLP_embedding_size
        self.hidden_layers_units = hidden_layers_units

        user_inputs = layers.Input(shape=(1,), dtype="int32", name="user_inputs")
        item_inputs = layers.Input(shape=(1,), dtype="int32", name="item_inputs")

        user_lookup = layers.IntegerLookup(vocabulary=unique_users, name="user_lookup")
        item_lookup = layers.IntegerLookup(vocabulary=unique_items, name="item_lookup")

        user_idxs = user_lookup(user_inputs)
        item_idxs = item_lookup(item_inputs)

        # GMF

        GMF_user_embeddings = layers.Flatten()(layers.Embedding(user_lookup.vocabulary_size(), self.GMF_embedding_size, initializers.RandomNormal(), name="GMF_user_embedding")(user_idxs))
        GMF_item_embeddings = layers.Flatten()(layers.Embedding(item_lookup.vocabulary_size(), self.GMF_embedding_size, initializers.RandomNormal(), name="GMF_item_embedding")(item_idxs))       

        GMF_outputs = layers.multiply([ GMF_user_embeddings, GMF_item_embeddings ])

        # MLP

        MLP_user_embeddings = layers.Flatten()(layers.Embedding(user_lookup.vocabulary_size(), self.MLP_embedding_size, initializers.RandomNormal(), name="MLP_user_embedding")(user_idxs))
        MLP_item_embeddings = layers.Flatten()(layers.Embedding(item_lookup.vocabulary_size(), self.MLP_embedding_size, initializers.RandomNormal(), name="MLP_item_embedding")(item_idxs))

        MLP_outputs = layers.concatenate([ MLP_user_embeddings, MLP_item_embeddings ])

        for i, units in enumerate(self.hidden_layers_units):

            MLP_outputs = layers.Dense(units, activation="relu", name=f"MLP_hidden_{i+1}")(MLP_outputs)

        # NeuMF

        outputs = layers.concatenate([ GMF_outputs, MLP_outputs ])

        outputs = layers.Dense(1, kernel_initializer=initializers.LecunUniform(), activation="sigmoid", name="outputs")(outputs)

        super().__init__(inputs=[ user_inputs, item_inputs ], outputs=outputs, **kwargs)

    def get_config(self):

        return {
            "unique_users": self.unique_users,
            "unique_items": self.unique_items,
            "GMF_embedding_size": self.GMF_embedding_size,
            "MLP_embedding_size": self.MLP_embedding_size,
            "hidden_layers_units": self.hidden_layers_units,
        }