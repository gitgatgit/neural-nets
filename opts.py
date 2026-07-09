import tensorflow as tf
import itertools
import numpy as np

def build_model(input_shape, num_layers, neurons, activation, dropout_rate, reg_rate, learning_rate):
    """
    Build a sequential Keras model using provided hyperparameters.
    - input_shape: shape of the input data (tuple)
    - num_layers: number of hidden dense layers
    - neurons: number of neurons per hidden layer
    - activation: activation function for hidden layers
    - dropout_rate: dropout rate (0.0 means no dropout)
    - reg_rate: L2 regularisation rate (0.0 means no regularisation)
    - learning_rate: learning rate for the Adam optimizer
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # Add hidden layers with optional L2 regularisation and dropout
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(neurons,
                                        activation=activation,
                                        kernel_regularizer=tf.keras.regularizers.l2(reg_rate) if reg_rate > 0 else None))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    # For demonstration we assume a 10-class classification problem.
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def grid_search_hyperparams(x_train, y_train, x_val, y_val, input_shape, param_grid, epochs=5):
    """
    Perform grid search over the provided hyperparameter grid.
    param_grid should be a dictionary with keys:
      - 'learning_rate'
      - 'num_layers'
      - 'neurons'
      - 'activation'
      - 'dropout_rate'
      - 'reg_rate'
      - 'batch_size'
    """
    best_val_acc = 0
    best_params = None
    best_model = None

    # Loop over all combinations of hyperparameters using itertools.product
    for lr, num_layers, neurons, activation, dropout_rate, reg_rate, batch_size in itertools.product(
            param_grid['learning_rate'],
            param_grid['num_layers'],
            param_grid['neurons'],
            param_grid['activation'],
            param_grid['dropout_rate'],
            param_grid['reg_rate'],
            param_grid['batch_size']):

        print(f"Testing: lr={lr}, layers={num_layers}, neurons={neurons}, activation={activation}, "
              f"dropout={dropout_rate}, reg={reg_rate}, batch_size={batch_size}")

        model = build_model(input_shape, num_layers, neurons, activation, dropout_rate, reg_rate, lr)

        # Train the model and record history
        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0)

        # Get the validation accuracy from the last epoch
        val_acc = history.history['val_accuracy'][-1]
        print(f"Validation accuracy: {val_acc:.4f}")

        # Save the model and hyperparameters if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'learning_rate': lr,
                'num_layers': num_layers,
                'neurons': neurons,
                'activation': activation,
                'dropout_rate': dropout_rate,
                'reg_rate': reg_rate,
                'batch_size': batch_size
            }
            best_model = model

    print("\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return best_model, best_params


# Example: Use the MNIST dataset (10-class handwritten digits)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess: flatten images and scale pixel values to [0, 1]
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

    # Create a validation split from the training data
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

input_shape = (28*28,)

    # Define the grid of hyperparameters to search over
param_grid = {
        'learning_rate': [0.001, 0.01],
        'num_layers': [1, 2],
        'neurons': [64, 128],
        'activation': ['relu', 'tanh'],
        'dropout_rate': [0.0, 0.2],
        'reg_rate': [0.0, 0.001],
        'batch_size': [32, 64]
}

    # Run the grid search
best_model, best_params = grid_search_hyperparams(x_train, y_train, x_val, y_val, input_shape, param_grid, epochs=5)

    # Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy of the best model: {test_acc:.4f}")
