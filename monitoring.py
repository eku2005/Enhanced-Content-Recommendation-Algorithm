import matplotlib.pyplot as plt

# Example function to monitor model performance
def plot_model_performance(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example function for continuous monitoring
def monitor_performance(model, test_data):
    users, items = np.nonzero(test_data)
    interactions = test_data[users, items]

    test_loss = model.evaluate(x=np.array([users, items]).T, y=interactions)
    print(f'Test Loss: {test_loss}')

# Call the monitoring functions
history = model.fit(x=np.array([users, items]).T, y=interactions, epochs=10, batch_size=256)
plot_model_performance(history)
monitor_performance(model, test)
