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


###################

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
    qubits = [cirq.GridQubit(0, i) for i in range(circ_size)]
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

    # adding rotation at the end to allow arbitrary measurement basis
    for i, q in enumerate(qubits):
        rx = sympy.Symbol(f'meas_rx{i}')
        ry = sympy.Symbol(f'meas_ry{i}')
        rz = sympy.Symbol(f'meas_rz{i}')
        circuit += [cirq.rx(rx)(q), cirq.ry(ry)(q), cirq.rz(rz)(q)]
        symbols += [rx, ry, rz]
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



def encode_qrac_3_circuit(q, b1, b2, b3):
    # Map bits to {-1, +1}
    x = 2*b1 - 1
    y = 2*b2 - 1
    z = 2*b3 - 1

    # Normalize
    norm = np.sqrt(x*x + y*y + z*z)
    x, y, z = x/norm, y/norm, z/norm

    # Convert to Bloch angles
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    # print(b1, b2, b3, "->", theta, phi)

    circuit = cirq.Circuit()
    circuit.append(cirq.ry(theta)(q))
    circuit.append(cirq.rz(phi)(q))
    return circuit


def encode_image_tuple_qrac_3(img):
    # img is 5x6 → 30 bits
    f = img.flatten()

    # Drop the first 3 bits (as per comment)
    # f = f[3:]          # now length = 27

    # Split into 12 triples
    triples = [(f[i], f[i+1], f[i+2]) for i in range(0, 36, 3)]

    # Create 12 qubits
    qubits = [cirq.GridQubit(0, i) for i in range(12)]

    # Build the circuit
    full = cirq.Circuit()

    for q, (b1, b2, b3) in zip(qubits, triples):
        full += encode_qrac_3_circuit(q, b1, b2, b3)

    return full




#################
print("qrac_6x6_36_12_qubits.py")
datafname = 'reduced_dataset_36pixels_QRAC.npz'
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


# picking N_samples from each class for training and testing
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

del_duplicates = False
if del_duplicates:
    # Remove duplicates
    X_flat = X_selected.reshape(len(X_selected), -1)
    X_unique, unique_indices = np.unique(X_flat, axis=0, return_index=True)
    y_unique = y_selected[unique_indices]
    print("Before:", len(X_selected))
    print("After:", len(X_unique))
    X_selected = X_unique
    y_selected = y_unique
    #########################

    ####### Balancing the classes ###############
    # Count samples per class
    n0 = np.sum(y_selected == 0)
    n1 = np.sum(y_selected == 1)

    # Choose balanced size
    N = min(n0, n1)

    # Randomly pick N samples from each class
    idx0 = np.random.choice(np.where(y_selected == 0)[0], N, replace=False)
    idx1 = np.random.choice(np.where(y_selected == 1)[0], N, replace=False)

    # Combine
    balanced_idx = np.concatenate([idx0, idx1])

    # Shuffle
    np.random.shuffle(balanced_idx)

    # Final balanced dataset
    X_selected = X_selected[balanced_idx]
    y_selected = y_selected[balanced_idx]

    N_samples = N



circuit_list = [encode_image_tuple_qrac_3(a) for a in X_selected]
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
model = build_model(num_layers=num_layers, shots=N_shot, circ_size = 12)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.05),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')]
)
model.summary()
batch_size = 200
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=batch_size,
    verbose=1
)


# ------------------------------------------------------------
# 6. Evaluate over the complete dataset
# ------------------------------------------------------------

data = np.load(datafname)
X = data['images']
y = data['labels']
# y = np.argmax(data['labels'], axis=1)   # shape becomes (60000,)


# keep only 0s and 1s
mask = (y == 0) | (y == 1)
X_filtered = X[mask]
y_filtered = y[mask]

circuit_list = [encode_image_tuple_qrac_3(a) for a in X_filtered]
circuit_tensor = tfq.convert_to_tensor(circuit_list)

circuit_np = circuit_tensor.numpy()   # convert TF tensor → NumPy array

logits = model.predict(circuit_np)
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
# export history
fname = f"classify_6x6_QRAC_Coding_DataQRACoptimizedV2_bs{batch_size}_Ns{N_samples}_Nh{N_shot}_Nl{num_layers}_dD{del_duplicates}.dat"
export_history(history, fname)
fname = f"classify_6x6_QRAC_Coding_DataQRACoptimizedV2_bs{batch_size}_Ns{N_samples}_Nh{N_shot}_Nl{num_layers}_dD{del_duplicates}.h5"
model.save_weights(fname)


