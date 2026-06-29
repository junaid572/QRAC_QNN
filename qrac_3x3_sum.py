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


def encode_image_tuple_qrac_3(img1, img2, img3):
    f1 = img1.flatten()
    f2 = img2.flatten()
    f3 = img3.flatten()

    # Create 16 qubits for 4×4 image
    qubits = [cirq.GridQubit(i, j) for i in range(4) for j in range(4)]

    full = cirq.Circuit()

    for q, b1, b2, b3 in zip(qubits, f1, f2, f3):
        full += encode_qrac_3_circuit(q, b1, b2, b3)

    return full




#################
print("qrac_3x3_sum.py")
datafname = 'reduced_dataset_9pixels_V1.npz'
print(f"Loading data from {datafname}...")
data = np.load(datafname)
X1 = data['images']
y1 = data['labels']

# keep only 0s and 1s
mask = (y1 == 0) | (y1 == 1)
X1_filtered = X1[mask]
y1_filtered = y1[mask]

datafname = 'reduced_dataset_9pixels_V2.npz'
print(f"Loading data from {datafname}...")
data = np.load(datafname)
X2 = data['images']
y2 = data['labels']

# keep only 0s and 1s
mask = (y2 == 0) | (y2 == 1)
X2_filtered = X2[mask]
y2_filtered = y2[mask]

datafname = 'reduced_dataset_9pixels_V3.npz'
print(f"Loading data from {datafname}...")
data = np.load(datafname)
X3 = data['images']
y3 = data['labels']

# keep only 0s and 1s
mask = (y3 == 0) | (y3 == 1)
X3_filtered = X3[mask]
y3_filtered = y3[mask]

# balance X1 across class 0 and 1
idx_0 = np.where(y1_filtered == 0)[0]
idx_1 = np.where(y1_filtered == 1)[0]
min_size = min(len(idx_0), len(idx_1))
idx_0_bal = np.random.choice(idx_0, min_size, replace=False)
idx_1_bal = np.random.choice(idx_1, min_size, replace=False)
balanced_idx = np.concatenate([idx_0_bal, idx_1_bal])
np.random.shuffle(balanced_idx)
X1_filtered = X1_filtered[balanced_idx]
y1_filtered = y1_filtered[balanced_idx]

# balance X2 across class 0 and 1
idx_0 = np.where(y2_filtered == 0)[0]
idx_1 = np.where(y2_filtered == 1)[0]
min_size = min(len(idx_0), len(idx_1))
idx_0_bal = np.random.choice(idx_0, min_size, replace=False)
idx_1_bal = np.random.choice(idx_1, min_size, replace=False)
balanced_idx = np.concatenate([idx_0_bal, idx_1_bal])
np.random.shuffle(balanced_idx)
X2_filtered = X2_filtered[balanced_idx]
y2_filtered = y2_filtered[balanced_idx]

# balance X3 across class 0 and 1
idx_0 = np.where(y3_filtered == 0)[0]
idx_1 = np.where(y3_filtered == 1)[0]
min_size = min(len(idx_0), len(idx_1))
idx_0_bal = np.random.choice(idx_0, min_size, replace=False)
idx_1_bal = np.random.choice(idx_1, min_size, replace=False)
balanced_idx = np.concatenate([idx_0_bal, idx_1_bal])
np.random.shuffle(balanced_idx)
X3_filtered = X3_filtered[balanced_idx]
y3_filtered = y3_filtered[balanced_idx]


# 4 x 4 image with block averaging to get 16 features
# X_features = X_filtered #np.array([extract_16_features(img) for img in X_filtered])
# X_features_bin = X_filtered# np.array([binarize_features(f) for f in X_features])



# making tuples of 3 images each, with label 1 if sum of labels is 2, else 0
num_tuples = 1000
tuple_feats = []
tuple_labels = []

for _ in range(num_tuples):
    idx = np.random.choice(len(X1_filtered), size=3, replace=True)
    feats = [X1_filtered[idx[0]], X2_filtered[idx[1]], X3_filtered[idx[2]]]     # shape (3, 16)
    labs  = [y1_filtered[idx[0]], y2_filtered[idx[1]], y3_filtered[idx[2]]]

    # label = 1 if sum of the labels is 2
    label = 1 if np.sum(labs) == 2 else 0

    tuple_feats.append(feats)
    tuple_labels.append(label)

tuple_feats = np.array(tuple_feats)     # (10000, 3, 16)
tuple_labels = np.array(tuple_labels)   # (10000,)


# balancing the data
# indices of each class
idx_0 = np.where(tuple_labels == 0)[0]
idx_1 = np.where(tuple_labels == 1)[0]

# choose the smaller class size
min_size = min(len(idx_0), len(idx_1))

print(f"Class 0: {len(idx_0)}, Class 1: {len(idx_1)}, Balanced size: {min_size}")

# sample equally
idx_0_bal = np.random.choice(idx_0, min_size, replace=False)
idx_1_bal = np.random.choice(idx_1, min_size, replace=False)

balanced_idx = np.concatenate([idx_0_bal, idx_1_bal])
np.random.shuffle(balanced_idx)

X_balanced = tuple_feats[balanced_idx]
y_balanced = tuple_labels[balanced_idx]

X_selected = X_balanced
y_selected = y_balanced

circuit_list = [encode_image_tuple_qrac_3(a, b, c) for (a,b,c) in X_selected]
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
# 6. Evaluate
# ------------------------------------------------------------

num_val_tuples = 10000
tuple_feats = []
tuple_labels = []

print(f"Testing on {num_val_tuples} new random tuples...")

for _ in range(num_val_tuples):
    idx = np.random.choice(len(X1_filtered), size=3, replace=True)
    feats = [X1_filtered[idx[0]], X2_filtered[idx[1]], X3_filtered[idx[2]]]     # shape (3, 16)
    labs  = [y1_filtered[idx[0]], y2_filtered[idx[1]], y3_filtered[idx[2]]]

    # label = 1 if sum of the labels is 2
    label = 1 if np.sum(labs) == 2 else 0

    tuple_feats.append(feats)
    tuple_labels.append(label)

tuple_feats = np.array(tuple_feats)     # (10000, 3, 16)
tuple_labels = np.array(tuple_labels)   # (10000,)


# balancing the data
# indices of each class
idx_0 = np.where(tuple_labels == 0)[0]
idx_1 = np.where(tuple_labels == 1)[0]

# choose the smaller class size
min_size = min(len(idx_0), len(idx_1))

# sample equally
idx_0_bal = np.random.choice(idx_0, min_size, replace=False)
idx_1_bal = np.random.choice(idx_1, min_size, replace=False)

balanced_idx = np.concatenate([idx_0_bal, idx_1_bal])
np.random.shuffle(balanced_idx)

X_balanced = tuple_feats[balanced_idx]
y_balanced = tuple_labels[balanced_idx]

X_selected = X_balanced
y_selected = y_balanced

circuit_list = [encode_image_tuple_qrac_3(a, b, c) for (a,b,c) in X_selected]
circuit_tensor = tfq.convert_to_tensor(circuit_list)

circuit_np = circuit_tensor.numpy()   # convert TF tensor → NumPy array

logits = model.predict(circuit_np)
preds = (logits > 0.0).astype(np.float32).flatten()

true = y_balanced.flatten()
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
fname = f"sum_2_3x3_9_qubit_bs{batch_size}_Ntuples{num_tuples}_Nh{N_shot}_Nl{num_layers}_QRACV1V2V3.dat"
export_history(history, fname)
fname = f"sum_2_3x3_9_qubit_bs{batch_size}_Ntuples{num_tuples}_Nh{N_shot}_Nl{num_layers}_QRACV1V2V3.h5"
model.save_weights(fname)


