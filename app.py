"""
MNIST-Style Digit Preprocessing + PyTorch Neural Network Studio
===================================================================
A Streamlit app that:
  1. Lets a user draw a digit on a canvas.
  2. Walks it through the classic MNIST preprocessing pipeline:
        Grayscale -> Binarize -> Crop -> Square Pad -> Resize -> Recenter
  3. Classifies the processed digit using a PyTorch neural network
     (Input Layer -> Hidden Layer -> Output Layer), built and run with tensors.
  4. Shows an educational, neuron-level diagram of that 3-layer network.
  5. Saves each user's drawn digits (as pixel data) into a CSV file named
     after that user, and thanks them by name after saving.

Run with:  streamlit run app.py

NOTE: Classification requires a trained model file named `mnist_pytorch.pt`
in the same folder as this app. Run `train_model.py` once to create it.
"""

import os
import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Optional: PyTorch for neural-network classification. The rest of the app
# (preprocessing, saving, export) still works even if PyTorch isn't installed.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="Digit Preprocessing + Neural Network Studio", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #E6F4FF; }
    section[data-testid="stSidebar"] { background-color: #CDEBFF; }
    .prediction-card {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6);
        color: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .prediction-digit { font-size: 64px; font-weight: 800; margin: 0; }
    .prediction-confidence { font-size: 18px; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DATA_DIR = "saved_digits"
MODEL_PATH = "mnist_pytorch.pt"


# --------------------------------------------------------------------------
# PyTorch model definition
# Must match train_model.py exactly, since we load a state_dict into it.
# Architecture: Input Layer (784) -> Hidden Layer (128) -> Output Layer (10)
# --------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class DigitClassifier(nn.Module):
        """A classic 3-layer feedforward neural network."""

        def __init__(self):
            super().__init__()
            self.input_to_hidden = nn.Linear(28 * 28, 128)   # Input layer -> Hidden layer
            self.hidden_to_output = nn.Linear(128, 10)         # Hidden layer -> Output layer
            self.relu = nn.ReLU()

        def forward(self, x):
            logits, _ = self.forward_with_activations(x)
            return logits

        def forward_with_activations(self, x):
            """Same forward pass as above, but also returns the hidden-layer
            activations so we can visualize the live forward flow neuron by
            neuron for whichever digit was just drawn."""
            x = x.view(x.size(0), -1)                       # Flatten image tensor to a 784-length vector
            hidden = self.relu(self.input_to_hidden(x))     # Hidden layer + ReLU activation
            logits = self.hidden_to_output(hidden)           # Output layer (raw scores per digit)
            return logits, hidden


# --------------------------------------------------------------------------
# STEP 1 — Grayscale & Binarization
# --------------------------------------------------------------------------
def to_grayscale_binary(rgba_image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Collapse the RGBA canvas to grayscale and binarize: background -> 0,
    digit stroke -> 255. Combines RGB brightness with the alpha channel since
    streamlit-drawable-canvas encodes strokes in both."""
    rgb = rgba_image[:, :, :3].astype(np.uint8)
    gray_from_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alpha = rgba_image[:, :, 3].astype(np.uint8)
    combined = cv2.bitwise_and(gray_from_rgb, alpha)
    _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
    return binary


# --------------------------------------------------------------------------
# STEP 2 — Bounding Box & Crop
# --------------------------------------------------------------------------
def crop_to_bounding_box(binary_img: np.ndarray, margin: int = 4):
    """Find the bounding box around all drawn (non-zero) pixels and crop
    tightly to it, with a small margin. Returns (cropped_img, found_flag)."""
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return None, False
    x, y, w, h = cv2.boundingRect(coords)
    H, W = binary_img.shape
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, W), min(y + h + margin, H)
    return binary_img[y0:y1, x0:x1], True


# --------------------------------------------------------------------------
# STEP 3 — Square Padding (centered, no distortion)
# --------------------------------------------------------------------------
def pad_to_square(cropped_img: np.ndarray, pad_frac: float = 0.2) -> np.ndarray:
    """Pad the cropped digit into a centered square, preserving aspect ratio."""
    h, w = cropped_img.shape
    side = max(h, w)
    border = int(side * pad_frac)
    side_padded = side + 2 * border
    square = np.zeros((side_padded, side_padded), dtype=cropped_img.dtype)
    y_off, x_off = (side_padded - h) // 2, (side_padded - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped_img
    return square


# --------------------------------------------------------------------------
# STEP 4 — Resize / Pixelation (area-based downsampling)
# --------------------------------------------------------------------------
def resize_pixelate(square_img: np.ndarray, target_size: int) -> np.ndarray:
    """Downscale using area-based interpolation — averages regions rather
    than sampling single points, avoiding jagged aliasing artifacts."""
    return cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# STEP 5 — Center of Mass Recentering
# --------------------------------------------------------------------------
def recenter_by_center_of_mass(img: np.ndarray) -> np.ndarray:
    """Shift the image so the intensity-weighted center of mass aligns with
    the grid's geometric center — the standard MNIST convention."""
    h, w = img.shape
    total_mass = img.sum()
    if total_mass == 0:
        return img.copy()
    y_indices, x_indices = np.indices((h, w))
    cy = (y_indices * img).sum() / total_mass
    cx = (x_indices * img).sum() / total_mass
    shift_y = int(round(h / 2.0 - cy))
    shift_x = int(round(w / 2.0 - cx))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


# --------------------------------------------------------------------------
# Visualization helper
# --------------------------------------------------------------------------
def show_matrix_heatmap(matrix: np.ndarray, title: str, annotate: bool = False):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(matrix, cmap="gray_r", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate and matrix.shape[0] <= 28:
        step = max(1, matrix.shape[0] // 14)
        for i in range(0, matrix.shape[0], step):
            for j in range(0, matrix.shape[1], step):
                val = int(matrix[i, j])
                ax.text(j, i, str(val), ha="center", va="center",
                         fontsize=4.5, color="red" if val < 128 else "black")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Model loading + prediction (PyTorch, tensor-based)
# --------------------------------------------------------------------------
@st.cache_resource
def load_nn_model():
    """Load the trained PyTorch model once and cache it across reruns."""
    if not TORCH_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None
    model = DigitClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict_digit(model, img_28: np.ndarray):
    """Convert the 28x28 digit into a PyTorch tensor, run it through the
    network, and return (predicted_digit, confidence, probs, input_vector,
    hidden_activations) — the last two are the actual tensor values at the
    input and hidden layers for this specific digit, used for the live
    forward-pass visualization."""
    x = torch.tensor(img_28, dtype=torch.float32) / 255.0   # normalize to [0, 1]
    x_input = x.view(1, 1, 28, 28)                            # (batch, channel, H, W) tensor shape
    with torch.no_grad():
        logits, hidden = model.forward_with_activations(x_input)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    predicted_digit = int(probs.argmax())
    confidence = float(probs[predicted_digit])
    input_vector = x.view(-1).numpy()          # the 784 flattened input values
    hidden_activations = hidden.squeeze(0).numpy()  # the 128 real hidden-neuron outputs
    return predicted_digit, confidence, probs, input_vector, hidden_activations


# --------------------------------------------------------------------------
# Neural network architecture diagram (actual neurons + connections)
# --------------------------------------------------------------------------
def draw_nn_diagram():
    """Draw the 3-layer network as circles (neurons) connected by lines,
    matching the Input -> Hidden -> Output architecture used for classification."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Representative neuron counts shown on screen (the real layers are
    # 784 / 128 / 10 — we draw a readable subset with "..." to indicate more).
    layers = [
        {"x": 0.5, "shown": 8, "real": 784, "label": "Input Layer\n(784 neurons)", "color": "#93C5FD"},
        {"x": 3.0, "shown": 10, "real": 128, "label": "Hidden Layer\n(128 neurons)", "color": "#3B82F6"},
        {"x": 5.5, "shown": 10, "real": 10, "label": "Output Layer\n(10 neurons: 0-9)", "color": "#1E3A8A"},
    ]

    positions = []
    for layer in layers:
        n = layer["shown"]
        ys = np.linspace(0.5, n - 0.5, n) if n > 1 else [n / 2]
        # Center around 0
        ys = np.array(ys) - np.mean(ys)
        coords = [(layer["x"], y) for y in ys]
        positions.append(coords)

        for (x, y) in coords:
            circle = plt.Circle((x, y), 0.18, facecolor=layer["color"], edgecolor="black", zorder=3)
            ax.add_patch(circle)

        # Label real neuron count under the drawn layer
        ax.text(layer["x"], min(ys) - 1.0, layer["label"], ha="center", va="top",
                 fontsize=9, fontweight="bold")

        # Digit labels on the output layer neurons
        if layer["real"] == 10:
            for i, (x, y) in enumerate(coords):
                ax.text(x + 0.35, y, str(i), ha="left", va="center", fontsize=8)

    # Draw connecting lines between consecutive layers (fully-connected look)
    for coords_a, coords_b in zip(positions[:-1], positions[1:]):
        for (xa, ya) in coords_a:
            for (xb, yb) in coords_b:
                ax.plot([xa, xb], [ya, yb], color="gray", linewidth=0.3, alpha=0.4, zorder=1)

    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-6.5, 6.5)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def draw_live_forward_diagram(input_vector: np.ndarray, hidden_activations: np.ndarray,
                               probs: np.ndarray, predicted_digit: int):
    """Draw the same 3-layer network, but color and size each shown neuron by
    its REAL activation value from this specific digit's forward pass —
    brighter/larger = more active. This is the 'live' view, generated fresh
    for every prediction, as opposed to the static structural diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Input layer: sample 10 pixels spread across the 784, so the picture
    # stays readable while still reflecting real (normalized 0-1) values.
    input_idx = np.linspace(0, len(input_vector) - 1, 10).astype(int)
    input_vals = input_vector[input_idx]

    # Hidden layer: show the 10 most-active neurons for this digit — the
    # ones that actually "fired" strongest, which is what makes this live
    # rather than a fixed structural sample.
    hidden_max = hidden_activations.max() if hidden_activations.max() > 0 else 1.0
    top_hidden_idx = np.argsort(hidden_activations)[-10:][::-1]
    hidden_vals = hidden_activations[top_hidden_idx] / hidden_max

    # Output layer: all 10 digit-probability neurons.
    output_vals = probs

    layers = [
        {"x": 0.5, "vals": input_vals, "label": "Input Layer\n(sampled pixels)"},
        {"x": 3.2, "vals": hidden_vals, "label": "Hidden Layer\n(top-10 firing neurons)"},
        {"x": 5.9, "vals": output_vals, "label": "Output Layer\n(digit probabilities)"},
    ]

    positions = []
    for layer in layers:
        n = len(layer["vals"])
        ys = np.linspace(0.5, n - 0.5, n) - np.mean(np.linspace(0.5, n - 0.5, n))
        coords = list(zip([layer["x"]] * n, ys))
        positions.append(coords)
        for (x, y), val in zip(coords, layer["vals"]):
            val = float(max(0.0, min(1.0, val)))
            color = plt.cm.Oranges(0.25 + 0.7 * val)   # brighter orange = more active
            radius = 0.15 + 0.15 * val                  # bigger circle = more active
            ax.add_patch(plt.Circle((x, y), radius, facecolor=color, edgecolor="black", zorder=3))
        ax.text(layer["x"], min(ys) - 1.0, layer["label"], ha="center", va="top",
                 fontsize=9, fontweight="bold")

    # Output layer: label each neuron with its digit + live probability,
    # highlighting the one the network actually predicted.
    for i, ((x, y), val) in enumerate(zip(positions[2], output_vals)):
        is_winner = (i == predicted_digit)
        ax.text(x + 0.35, y, f"{i}: {val * 100:.0f}%", ha="left", va="center",
                 fontsize=8, color=("#166534" if is_winner else "black"),
                 fontweight=("bold" if is_winner else "normal"))

    # Connections: line brightness reflects the activation of the neuron
    # it's coming FROM, so you can visually trace the strongest signal path.
    for coords_a, vals_a, coords_b in zip(positions[:-1],
                                           [layers[0]["vals"], layers[1]["vals"]],
                                           positions[1:]):
        vmax = max(np.max(np.abs(vals_a)), 1e-8)
        for (xa, ya), va in zip(coords_a, vals_a):
            strength = float(min(1.0, abs(va) / vmax))
            for (xb, yb) in coords_b:
                ax.plot([xa, xb], [ya, yb], color="orangered",
                         linewidth=0.3 + 0.5 * strength, alpha=0.1 + 0.4 * strength, zorder=1)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-6.5, 6.5)
    ax.axis("off")
    ax.set_title(f"Live Forward Pass → Predicted Digit: {predicted_digit}", fontsize=12, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)


NN_LAYER_EXPLANATIONS = [
    ("Input Layer (784 neurons)", "Every pixel of the 28x28 preprocessed digit becomes one input neuron, converted into a tensor."),
    ("Hidden Layer (128 neurons)", "Learns weighted combinations of pixel patterns, passed through a ReLU activation to capture shapes like curves and loops."),
    ("Output Layer (10 neurons)", "One neuron per digit (0-9). A softmax turns their scores into probabilities — the highest one is the prediction."),
]


# --------------------------------------------------------------------------
# Save helper — one CSV file per user, named after them
# --------------------------------------------------------------------------
def save_digit_to_user_file(username: str, recentered_img: np.ndarray,
                             true_label, predicted_digit, confidence, resolution) -> str:
    """Append this drawn digit's pixel values + metadata as one row to a CSV
    file named after the user (e.g. saved_digits/pankaj.csv). Creates the
    file with a header on the first save."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    safe_name = "".join(c for c in username if c.isalnum() or c in ("_", "-")) or "anonymous"
    file_path = os.path.join(BASE_DATA_DIR, f"{safe_name}.csv")

    row = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "true_label": true_label,
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "resolution": resolution,
    }
    flat_pixels = recentered_img.flatten().tolist()
    for i, val in enumerate(flat_pixels):
        row[f"pixel_{i}"] = int(val)

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(file_path)
    df_row.to_csv(file_path, mode="a", header=write_header, index=False)
    return file_path


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("🎛️ Controls")

username = st.sidebar.text_input("👤 Username", value="guest",
                                  help="Your drawn digits are saved to a CSV file named after you.")

stroke_width = st.sidebar.slider("🎨 Stroke width", min_value=5, max_value=25, value=15,
                                  help="How thick your pen stroke is on the canvas.")
st.sidebar.caption("Thicker strokes are bolder but less detailed.")

target_size = st.sidebar.select_slider("🔳 Target resolution", options=[14, 28, 56], value=28,
                                        help="Final grid size the digit is pixelated to. 28x28 matches MNIST.")

bin_threshold = st.sidebar.slider("⚫⚪ Binarization threshold", min_value=1, max_value=100, value=20,
                                   help="Pixels above this intensity are treated as digit ink.")

pad_fraction = st.sidebar.slider("⬛ Square padding (border %)", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
                                  help="Extra blank border added when squaring the digit.")

if st.sidebar.button("🗑️ Clear Canvas"):
    st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1

st.sidebar.markdown("---")
show_nn_arch = st.sidebar.toggle("🧠 Show Neural Network Architecture", value=False,
                                  help="Displays the 3-layer network diagram + explanation.")

st.sidebar.markdown("---")
batch_mode = st.sidebar.toggle("📦 Batch Mode", value=False,
                                help="Collect multiple drawn digits before saving them all together.")

st.sidebar.caption("Pipeline: Grayscale → Binarize → Crop → Square Pad → Resize → Recenter → Predict")


# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------
if "batch_digits" not in st.session_state:
    st.session_state["batch_digits"] = []


# --------------------------------------------------------------------------
# Main title
# --------------------------------------------------------------------------
st.title("✏️ MNIST-Style Digit Preprocessing + Neural Network Studio")
st.caption(
    "Draw a digit, watch it move through the MNIST preprocessing pipeline, "
    "then see a PyTorch neural network classify it in real time."
)

if show_nn_arch:
    st.markdown("### 🧠 Neural Network Architecture")
    draw_nn_diagram()
    cols = st.columns(len(NN_LAYER_EXPLANATIONS))
    for col, (name, explanation) in zip(cols, NN_LAYER_EXPLANATIONS):
        with col:
            st.markdown(f"**{name}**")
            st.caption(explanation)
    st.caption(
        "Note on accuracy: this network typically reaches ~97-98% accuracy on the standard MNIST "
        "test set. No model can honestly promise perfect accuracy on arbitrary handwriting — "
        "messy or ambiguous strokes will occasionally confuse any classifier."
    )
    st.markdown("---")

left_col, right_col = st.columns([1, 1.6])

with left_col:
    st.subheader("Draw a digit")
    canvas_key = f"canvas_{st.session_state.get('canvas_key', 0)}"
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=canvas_key,
        return_image_data=True,  # required since streamlit-drawable-canvas v0.10.0 — image_data is opt-in
    )
    st.caption("Black canvas, white stroke — matches MNIST's ink-on-background convention.")

    true_label = st.selectbox(
        "Optional: true label (for dataset building)",
        options=["(none)"] + [str(i) for i in range(10)],
        help="If you know what digit this is, tag it — useful for building a labeled training set."
    )

    if batch_mode:
        st.info(f"📦 Batch Mode ON — {len(st.session_state['batch_digits'])} digit(s) collected so far.")

with right_col:
    st.subheader("Pipeline visualizations")
    st.caption("Each tile shows the digit after one processing step — read left-to-right, top-to-bottom.")

    if canvas_result.image_data is None:
        st.info("Draw a digit on the left to see the processing pipeline.")
    else:
        raw_rgba = canvas_result.image_data.astype(np.uint8)
        binary_img = to_grayscale_binary(raw_rgba, threshold=bin_threshold)

        if binary_img.max() == 0:
            st.warning("⚠️ Canvas is empty — please draw a digit first.")
        else:
            cropped_img, found = crop_to_bounding_box(binary_img)

            if not found or cropped_img.size == 0:
                st.warning("⚠️ Couldn't detect a digit stroke. Try drawing again.")
            else:
                squared_img = pad_to_square(cropped_img, pad_frac=pad_fraction)
                pixelated_img = resize_pixelate(squared_img, target_size)
                recentered_img = recenter_by_center_of_mass(pixelated_img)

                # Always produce a 28x28 version for classification, regardless
                # of the chosen visualization resolution.
                pixelated_28 = resize_pixelate(squared_img, 28)
                recentered_28 = recenter_by_center_of_mass(pixelated_28)

                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    show_matrix_heatmap(255 - binary_img, "1. Grayscale + Binarized")
                    st.caption("Color dropped; every pixel becomes pure background or ink.")
                with r1c2:
                    show_matrix_heatmap(255 - squared_img, "2-3. Cropped + Square-Padded")
                    st.caption("Empty space trimmed, then bordered into a centered square.")

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    show_matrix_heatmap(255 - pixelated_img, f"4. Pixelated ({target_size}x{target_size})",
                                         annotate=(target_size <= 28))
                    st.caption("Shrunk using area-averaging — this creates the pixelated look.")
                with r2c2:
                    show_matrix_heatmap(255 - recentered_img, f"5. Recentered ({target_size}x{target_size})",
                                         annotate=(target_size <= 28))
                    st.caption("Center of mass shifted to the grid's middle — the MNIST convention.")

                st.markdown("---")
                st.subheader(f"Final {target_size}×{target_size} pixel matrix")
                st.caption("The numeric grid a classifier actually sees — each cell is 0 (black) to 255 (white).")
                st.dataframe(recentered_img, use_container_width=True)

                with st.expander("Show flattened tensor (as fed to the network)"):
                    flat_vector = (recentered_img.astype(np.float32) / 255.0).flatten()
                    st.write(f"Tensor length: {flat_vector.shape[0]}")
                    st.code(np.array2string(flat_vector, precision=3, threshold=50))

                # ------------------------------------------------------------
                # Neural Network Prediction
                # ------------------------------------------------------------
                st.markdown("---")
                st.subheader("🧠 Neural Network Prediction")

                model = load_nn_model()
                predicted_digit, confidence, probs = None, None, None

                if not TORCH_AVAILABLE:
                    st.warning("PyTorch isn't installed, so prediction is unavailable. "
                               "Run `pip install torch torchvision` to enable it.")
                elif model is None:
                    st.warning(f"No trained model found at `{MODEL_PATH}`. "
                               "Run `train_model.py` once to create it, then restart this app.")
                else:
                    predicted_digit, confidence, probs, input_vector, hidden_activations = predict_digit(
                        model, recentered_28
                    )
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <p class="prediction-digit">{predicted_digit}</p>
                            <p class="prediction-confidence">Confidence: {confidence * 100:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Show full probability breakdown"):
                        prob_df = pd.DataFrame({"digit": list(range(10)), "probability": probs})
                        st.bar_chart(prob_df.set_index("digit"))

                    st.markdown("#### 🔴 Live Forward Pass")
                    st.caption(
                        "This diagram is generated fresh for THIS digit — neuron brightness and size "
                        "reflect the network's real activation values as they flow Input → Hidden → Output."
                    )
                    draw_live_forward_diagram(input_vector, hidden_activations, probs, predicted_digit)

                # ------------------------------------------------------------
                # Save / Export
                # ------------------------------------------------------------
                st.markdown("---")
                st.subheader("💾 Save & Export")
                st.caption(f"Your saved digits are stored in `saved_digits/{username}.csv`.")

                label_value = None if true_label == "(none)" else int(true_label)

                save_col1, save_col2 = st.columns(2)
                with save_col1:
                    if batch_mode:
                        if st.button("➕ Add Current Digit to Batch"):
                            st.session_state["batch_digits"].append({
                                "processed_img": recentered_img.copy(),
                                "true_label": label_value,
                                "predicted_digit": predicted_digit,
                                "confidence": confidence,
                                "resolution": target_size,
                            })
                            st.success("Added to batch.")
                    else:
                        if st.button("💾 Save Digit"):
                            file_path = save_digit_to_user_file(
                                username, recentered_img, label_value,
                                predicted_digit, confidence, target_size
                            )
                            st.success(f"✅ Thank you, {username}! Your digit has been saved to `{file_path}`.")

                with save_col2:
                    png_buffer = io.BytesIO()
                    Image.fromarray(recentered_img.astype(np.uint8), mode="L").save(png_buffer, format="PNG")
                    st.download_button("⬇️ Download Digit (PNG)", data=png_buffer.getvalue(),
                                        file_name="digit.png", mime="image/png")

                safe_name = "".join(c for c in username if c.isalnum() or c in ("_", "-")) or "anonymous"
                user_file_path = os.path.join(BASE_DATA_DIR, f"{safe_name}.csv")
                if os.path.exists(user_file_path):
                    with open(user_file_path, "rb") as f:
                        st.download_button(
                            f"⬇️ Download All of {username}'s Saved Digits (CSV)",
                            data=f.read(),
                            file_name=f"{safe_name}.csv",
                            mime="text/csv",
                        )

# --------------------------------------------------------------------------
# Batch mode panel
# --------------------------------------------------------------------------
if batch_mode and st.session_state["batch_digits"]:
    st.markdown("---")
    st.subheader(f"📦 Batch ({len(st.session_state['batch_digits'])} digits collected)")

    thumb_cols = st.columns(min(8, len(st.session_state["batch_digits"])))
    for i, entry in enumerate(st.session_state["batch_digits"]):
        with thumb_cols[i % len(thumb_cols)]:
            st.image(255 - entry["processed_img"], width=60,
                     caption=str(entry["predicted_digit"]) if entry["predicted_digit"] is not None else "?")

    batch_col1, batch_col2 = st.columns(2)
    with batch_col1:
        if st.button("💾 Save Entire Batch"):
            for entry in st.session_state["batch_digits"]:
                save_digit_to_user_file(
                    username, entry["processed_img"], entry["true_label"],
                    entry["predicted_digit"], entry["confidence"], entry["resolution"]
                )
            st.success(f"✅ Thank you, {username}! Saved {len(st.session_state['batch_digits'])} digits.")
            st.session_state["batch_digits"] = []
    with batch_col2:
        if st.button("🗑️ Clear Batch"):
            st.session_state["batch_digits"] = []
            st.success("Batch cleared.")
