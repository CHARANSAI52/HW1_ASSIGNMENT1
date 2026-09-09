"""
Neural Network & Deep Learning Assignment
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Create output directories if they do not already exist.
PLOTS_DIR = Path("plots")
LOG_DIR = Path("logs/fit")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set seeds so the demonstration is more reproducible.
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# Part II - Task 1: Tensor Manipulations & Reshaping
# ============================================================

print("\n" + "=" * 60)
print("TASK 1: TENSOR MANIPULATIONS & RESHAPING")
print("=" * 60)

# 1. Create a random tensor with shape (4, 6).
tensor = tf.random.uniform((4, 6))

# 2. Find and print the rank and shape.
print("Original tensor:")
print(tensor.numpy())
print("Original rank:", tf.rank(tensor).numpy())
print("Original shape:", tensor.shape)

# 3. Reshape to (2, 3, 4), then transpose to (3, 2, 4).
reshaped = tf.reshape(tensor, (2, 3, 4))
transposed = tf.transpose(reshaped, perm=[1, 0, 2])

print("\nAfter reshaping to (2, 3, 4):")
print("Rank:", tf.rank(reshaped).numpy())
print("Shape:", reshaped.shape)

print("\nAfter transposing to (3, 2, 4):")
print("Rank:", tf.rank(transposed).numpy())
print("Shape:", transposed.shape)

# 4. Broadcast a tensor of shape (1, 4) across the last dimension.
small_tensor = tf.random.uniform((1, 4))

# A (1, 4) tensor broadcasts across the first two dimensions
# of a compatible (3, 2, 4) tensor.
small_broadcast = tf.broadcast_to(small_tensor, tf.shape(transposed))
added_tensor = transposed + small_tensor

print("\nSmaller tensor shape:", small_tensor.shape)
print("Broadcasted tensor shape:", small_broadcast.shape)
print("Added tensor shape:", added_tensor.shape)

# Broadcasting allows TensorFlow to conceptually expand dimensions
# of size 1 without manually copying the data. Dimensions are
# compatible when they are equal or when one of them is 1.


# ============================================================
# Part II - Task 2: Loss Functions & Hyperparameter Tuning
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: LOSS FUNCTIONS")
print("=" * 60)

# True one-hot class labels for three samples and three classes.
y_true = tf.constant(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=tf.float32,
)

# Initial predictions. Each row sums to 1, so these are valid
# probability distributions for categorical cross-entropy.
y_pred_1 = tf.constant(
    [
        [0.80, 0.10, 0.10],
        [0.10, 0.70, 0.20],
        [0.10, 0.20, 0.70],
    ],
    dtype=tf.float32,
)

# Slightly modified predictions.
y_pred_2 = tf.constant(
    [
        [0.70, 0.20, 0.10],
        [0.15, 0.65, 0.20],
        [0.10, 0.25, 0.65],
    ],
    dtype=tf.float32,
)

# Mean squared error.
mse = tf.keras.losses.MeanSquaredError()

# Categorical cross-entropy.
cce = tf.keras.losses.CategoricalCrossentropy()

mse_1 = float(mse(y_true, y_pred_1).numpy())
cce_1 = float(cce(y_true, y_pred_1).numpy())
mse_2 = float(mse(y_true, y_pred_2).numpy())
cce_2 = float(cce(y_true, y_pred_2).numpy())

print(f"Prediction set 1 - MSE: {mse_1:.6f}")
print(f"Prediction set 1 - CCE: {cce_1:.6f}")
print(f"Prediction set 2 - MSE: {mse_2:.6f}")
print(f"Prediction set 2 - CCE: {cce_2:.6f}")

# Plot the two loss functions for both prediction sets.
labels = ["MSE", "Categorical Cross-Entropy"]
losses_1 = [mse_1, cce_1]
losses_2 = [mse_2, cce_2]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, losses_1, width, label="Prediction Set 1")
plt.bar(x + width / 2, losses_2, width, label="Prediction Set 2")
plt.xticks(x, labels)
plt.ylabel("Loss")
plt.title("MSE vs. Categorical Cross-Entropy")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loss_comparison.png", dpi=150)
plt.close()


# ============================================================
# Shared MNIST preprocessing
# ============================================================

print("\n" + "=" * 60)
print("LOADING MNIST")
print("=" * 60)

# Load the handwritten-digit MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1].
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reserve part of the original training set for validation.
validation_size = 10000
x_validation = x_train[-validation_size:]
y_validation = y_train[-validation_size:]
x_train_small = x_train[:-validation_size]
y_train_small = y_train[:-validation_size]

# Convert integer labels to one-hot vectors for categorical
# cross-entropy.
y_train_cat = tf.keras.utils.to_categorical(y_train_small, 10)
y_validation_cat = tf.keras.utils.to_categorical(y_validation, 10)


def build_mnist_model(optimizer):
    """Build and compile a simple fully connected MNIST classifier."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            # Flatten converts each 28x28 image into a vector.
            tf.keras.layers.Flatten(),
            # Hidden dense layer learns nonlinear image features.
            tf.keras.layers.Dense(128, activation="relu"),
            # Dropout is intentionally omitted so optimizer trends
            # can be compared directly in this simple experiment.
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# Part II - Task 3: Adam vs SGD
# ============================================================

print("\n" + "=" * 60)
print("TASK 3: TRAINING WITH ADAM AND SGD")
print("=" * 60)

# Adam generally adapts the learning rate for each parameter.
adam_model = build_mnist_model(
    tf.keras.optimizers.Adam(learning_rate=0.001)
)

# SGD updates parameters using a fixed learning rate in this setup.
sgd_model = build_mnist_model(
    tf.keras.optimizers.SGD(learning_rate=0.01)
)

# Train both models for the same number of epochs.
adam_history = adam_model.fit(
    x_train_small,
    y_train_cat,
    validation_data=(x_validation, y_validation_cat),
    epochs=5,
    batch_size=128,
    verbose=1,
)

sgd_history = sgd_model.fit(
    x_train_small,
    y_train_cat,
    validation_data=(x_validation, y_validation_cat),
    epochs=5,
    batch_size=128,
    verbose=1,
)

print("\nFinal Adam training accuracy:",
      f"{adam_history.history['accuracy'][-1]:.4f}")
print("Final Adam validation accuracy:",
      f"{adam_history.history['val_accuracy'][-1]:.4f}")
print("Final SGD training accuracy:",
      f"{sgd_history.history['accuracy'][-1]:.4f}")
print("Final SGD validation accuracy:",
      f"{sgd_history.history['val_accuracy'][-1]:.4f}")

# Plot training accuracy.
epochs = range(1, 6)

plt.figure(figsize=(8, 5))
plt.plot(epochs, adam_history.history["accuracy"], marker="o", label="Adam - Training")
plt.plot(epochs, adam_history.history["val_accuracy"], marker="o", label="Adam - Validation")
plt.plot(epochs, sgd_history.history["accuracy"], marker="o", label="SGD - Training")
plt.plot(epochs, sgd_history.history["val_accuracy"], marker="o", label="SGD - Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy: Adam vs. SGD")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "adam_vs_sgd_accuracy.png", dpi=150)
plt.close()


# ============================================================
# Part II - Task 4: TensorBoard
# ============================================================

print("\n" + "=" * 60)
print("TASK 4: TENSORBOARD")
print("=" * 60)

# Use a unique TensorBoard subdirectory for this run.
tensorboard_log_dir = LOG_DIR / "mnist_run"

tensorboard_model = build_mnist_model(
    tf.keras.optimizers.Adam(learning_rate=0.001)
)

# TensorBoard callback records loss and accuracy for training and
# validation at each epoch.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=str(tensorboard_log_dir),
    histogram_freq=1,
)

tensorboard_model.fit(
    x_train_small,
    y_train_cat,
    validation_data=(x_validation, y_validation_cat),
    epochs=5,
    batch_size=128,
    callbacks=[tensorboard_callback],
    verbose=1,
)

print("\nTensorBoard logs saved to:", tensorboard_log_dir)
print("Launch TensorBoard with:")
print("tensorboard --logdir logs/fit")


# ============================================================
# Optional evaluation
# ============================================================

test_loss, test_accuracy = tensorboard_model.evaluate(
    x_test,
    tf.keras.utils.to_categorical(y_test, 10),
    verbose=0,
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nAll programming tasks completed.")
print(f"Plots are saved in: {PLOTS_DIR.resolve()}")
print(f"TensorBoard logs are saved in: {LOG_DIR.resolve()}")
