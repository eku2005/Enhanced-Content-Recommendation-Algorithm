import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Example dataset preparation
def prepare_dataset(user_interactions):
    user_interactions['interaction'] = 1
    data = user_interactions.pivot(index='user_id', columns='content_id', values='interaction').fillna(0)
    return data

data = prepare_dataset(user_interactions)

# Split data into training and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Define a simple collaborative filtering model using TensorFlow
class CFModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(CFModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        dot_product = tf.reduce_sum(user_vector * item_vector, axis=1)
        return dot_product

# Example function to train the model
def train_cf_model(data):
    num_users, num_items = data.shape
    model = CFModel(num_users, num_items, embedding_size=50)

    model.compile(optimizer='adam', loss='mse')
    
    users, items = np.nonzero(data)
    interactions = data[users, items]

    model.fit(x=np.array([users, items]).T, y=interactions, epochs=10, batch_size=256)
    
    return model

model = train_cf_model(train)
