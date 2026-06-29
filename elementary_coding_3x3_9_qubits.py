import importlib, pkg_resources
importlib.reload(pkg_resources)

# Keras 2 must be selected before importing TensorFlow or TensorFlow Quantum:
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import qutip as qt
import cv2
# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
from sklearn.model_selection import train_test_split



########################
class QuantumParamLayer(tf.keras.layers.Layer):
    """Holds a trainable vector of quantum parameters and tiles it to batch size."""
    def __init__(self, num_params, **kwargs):
        super().__init__(**kwargs)
        self.num_params = num_params

    def build(self, input_shape):
        self.params = self.add_weight(
            shape=(1, self.num_params),
            initializer='random_uniform',
            trainable=True,
            name='quantum_params'
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs is the circuit tensor, used only to get batch size
        batch_size = tf.shape(inputs)[0]
        return tf.tile(self.params, [batch_size, 1])

# ------------------------------------------------------------
# 4. Build the QNN with directly trainable parameters
# ------------------------------------------------------------
def build_model(num_layers=2, shots=100, circ_size = 16):
    qubits = [cirq.GridQubit(0, i) for i in range(circ_size*circ_size)]
    circuit = cirq.Circuit()
    symbols = []

    # print(circ_size, qubits)

    # Variational layers
    for layer in range(num_layers):
        for i, q in enumerate(qubits):
            rx = sympy.Symbol(f'l{layer}_rx{i}')
            ry = sympy.Symbol(f'l{layer}_ry{i}')
            rz = sympy.Symbol(f'l{layer}_rz{i}')
            circuit += [cirq.rx(rx)(q), cirq.ry(ry)(q), cirq.rz(rz)(q)]
            symbols += [rx, ry, rz]

        # Entangle ALL qubits
        for i in range(len(qubits)):
            circuit.append(cirq.CNOT(qubits[i], qubits[(i+1) % len(qubits)]))

    # Measure Z on all qubits
    observables = [cirq.Z(q) for q in qubits]

    # Keras model
    circuits_in = tf.keras.Input(shape=(), dtype=tf.string, name='circuits')
    param_values = QuantumParamLayer(len(symbols))(circuits_in)

    pqc = tfq.layers.ControlledPQC(circuit, observables, repetitions=shots)
    quantum_out = pqc([circuits_in, param_values])

    x = tf.keras.layers.Dense(10, activation='relu')(quantum_out)
    # x = tf.keras.layers.Dense(32, activation='relu')(x)
    logits = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=circuits_in, outputs=logits)


def export_history(history, filename="data.dat"):
    # Extract values
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy']

    epochs = np.arange(1, len(train_loss) + 1)

    # Stack columns
    data = np.column_stack([epochs, train_loss, val_loss, train_acc, val_acc])

    # Column labels (TikZ-friendly: no underscores)
    header = "epoch trainloss valloss trainacc valacc"

    # Save file
    np.savetxt(filename, data, header=header, comments='', fmt="%.6f")

    print(f"Saved training history to {filename}")

def angle_encode_image(img):
    # img is 3x3 with values in [0, 1]
    qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    circuit = cirq.Circuit()

    flat = img.flatten()

    for val, q in zip(flat, qubits):
        theta = val * np.pi  # scale to [0, π]
        circuit.append(cirq.ry(theta)(q))

    return circuit

def encode_image(img):
    # Elementary encoding of a 3x3 binary image into a quantum circuit
    qubits = [cirq.GridQubit(i, j) for i in range(3) for j in range(3)]
    circuit = cirq.Circuit()

    flat = img.flatten()

    for bit, q in zip(flat, qubits):
        if bit:
            circuit.append(cirq.X(q))

    return circuit
##########################

print("elementary_coding_3x3_9_qubits.py")
datafname = 'reduced_dataset_9pixels_V2.npz'
print(f"Loading data from {datafname}...")
data = np.load(datafname)
X = data['images']
y = data['labels']
# y = np.argmax(data['labels'], axis=1)   # shape becomes (60000,)


# keep only 0s and 1s
mask = (y == 0) | (y == 1)
X_filtered = X[mask]
y_filtered = y[mask]

# 3 x 3 image with block averaging to get 9 features
# X_features = np.array([extract_9_features(img) for img in X_filtered])
X_features_bin = X_filtered #np.array([binarize_features(f, threshold = 0.2) for f in X_features])



# -------------------------------
# 5. Select N samples per class
# -------------------------------
N_samples = 300
class_0_indices = np.where(y_filtered == 0)[0][:N_samples]
class_1_indices = np.where(y_filtered == 1)[0][:N_samples]
selected_indices = np.concatenate([class_0_indices, class_1_indices])
X_selected = X_features_bin[selected_indices]
y_selected = y_filtered[selected_indices]
# shuffle the samples
indices = np.random.permutation(len(X_selected))
X_selected = X_selected[indices]
y_selected = y_selected[indices]  


circuit_list = [encode_image(a) for a in X_selected]
circuit_tensor = tfq.convert_to_tensor(circuit_list)

circuit_np = circuit_tensor.numpy()   # convert TF tensor → NumPy array


X_train, X_val, y_train, y_val = train_test_split(
    circuit_np,
    y_selected,
    test_size=0.2,
    shuffle=True,
    stratify=y_selected,   # keeps class balance
    random_state=42
)


N_shot = 1000
num_layers = 4
model = build_model(num_layers=num_layers, shots=N_shot, circ_size = 3)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.05),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')]
)
model.summary()

batch_size = 300

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=batch_size,
    verbose=1
)


circuit_list = [encode_image(a) for a in X_filtered]
circuit_tensor = tfq.convert_to_tensor(circuit_list)

circuit_np = circuit_tensor.numpy()   # convert TF tensor → NumPy array
logits = model.predict(circuit_np, batch_size=batch_size)
preds = (logits > 0.0).astype(np.float32).flatten()

true = y_filtered.flatten()
print("Validation accuracy:", np.mean(preds == true))

tp = np.sum((preds == 1) & (true == 1))
tn = np.sum((preds == 0) & (true == 0))
fp = np.sum((preds == 1) & (true == 0))
fn = np.sum((preds == 0) & (true == 1))

conf_mat = np.array([[tn, fp],
                     [fn, tp]])

print("Confusion matrix:")
print(conf_mat)



fname = f"classify_3x3_elementary_Coding_DataV2_bs{batch_size}_Ns{N_samples}_Nh{N_shot}_Nl{num_layers}.dat"

export_history(history, fname)

fname = f"classify_3x3_elementary_Coding_DataV2_bs{batch_size}_Ns{N_samples}_Nh{N_shot}_Nl{num_layers}.h5"
model.save_weights(fname)