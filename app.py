"""
Streamlit App — WGAN CIFAR-10 Image Generator
Run with:  streamlit run app.py
"""

import os
import io
import time
import torch
import numpy as np
import streamlit as st
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# ─── import the model definition ─────────────────────────────────────────── #
from wgan_cifar10 import Generator, Critic, LATENT_DIM, device, \
                          get_cifar10_loader, CHECKPOINT_DIR, SAMPLE_DIR

# ─────────────────────────── Page Config ────────────────────────────────── #
st.set_page_config(
    page_title="WGAN · CIFAR-10 Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f4f8;
    text-shadow: 0 0 30px rgba(100, 200, 255, 0.4);
    margin-bottom: 0;
}

.sub-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #8ca0b0;
    margin-top: 4px;
}

.metric-card {
    background: linear-gradient(135deg, #1a2332 0%, #0d1a26 100%);
    border: 1px solid #2a3f52;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #64c8ff;
    font-weight: 700;
}

.metric-label {
    font-size: 0.75rem;
    color: #7a90a0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.info-pill {
    display: inline-block;
    background: rgba(100, 200, 255, 0.12);
    border: 1px solid rgba(100, 200, 255, 0.25);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    color: #64c8ff;
    margin: 3px;
    font-family: 'Space Mono', monospace;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #64c8ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2a3f52;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.stButton > button {
    background: linear-gradient(135deg, #1e6fa8 0%, #0d4a73 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2580bf 0%, #1460a0 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(30, 111, 168, 0.4);
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a26 0%, #080f18 100%);
    border-right: 1px solid #1e3045;
}

.stProgress .st-bo {
    background-color: #64c8ff;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Helpers ────────────────────────────────────── #

@st.cache_resource
def load_generator(checkpoint_path):
    G = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, ckpt


def tensor_grid_to_pil(tensor_batch, nrow=8):
    grid = torchvision.utils.make_grid(
        tensor_batch.cpu(), nrow=nrow,
        normalize=True, value_range=(-1, 1), padding=2
    )
    np_grid = grid.permute(1, 2, 0).numpy()
    np_grid = (np_grid * 255).astype(np.uint8)
    return Image.fromarray(np_grid)


def get_available_checkpoints():
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    return sorted(files)


def get_sample_images():
    if not os.path.exists(SAMPLE_DIR):
        return []
    files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")]
    return sorted(files)


# ───────────────────────────── Sidebar ──────────────────────────────────── #
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.5rem">⚡ WGAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">CIFAR-10 · Image Generator</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Model Config</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
      <span class="info-pill">Latent dim: 128</span>
      <span class="info-pill">Critic steps: 5</span>
      <span class="info-pill">Clip: ±0.01</span>
      <span class="info-pill">LR: 5e-5</span>
      <span class="info-pill">RMSprop</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    **Generator** — DCGAN-style  
    `z(128)` → `FC` → `256×8×8`  
    → `Upsample + Conv` → `128×16×16`  
    → `Upsample + Conv` → `64×32×32`  
    → `Conv` → `3×32×32` (Tanh)

    **Critic** — No sigmoid  
    `3×32×32` → 4 strided convs  
    → `FC` → raw score
    """)

    st.markdown("---")
    st.markdown('<div class="section-header">Loss (Wasserstein)</div>', unsafe_allow_html=True)
    st.markdown(r"""
    **Critic:** $\min_w \ \mathbb{E}[C(fake)] - \mathbb{E}[C(real)]$  
    **Generator:** $\min_\theta \ -\mathbb{E}[C(G(z))]$  
    **Constraint:** $w \in [-c, c]$ (weight clipping)
    """)


# ───────────────────────────── Main Area ────────────────────────────────── #
st.markdown('<div class="main-title">🎨 WGAN · CIFAR-10 Image Generator</div>', unsafe_allow_html=True)
st.markdown("")

# ✅ ONLY 2 TABS NOW
tab1, tab3 = st.tabs([
    "🖼️  Generate Images",
    "📊  Training Progress",
])


# ══════════════════════ Tab 1: Generate Images ═══════════════════════════ #
with tab1:
    col_ctrl, col_out = st.columns([1, 2])

    with col_ctrl:
        st.markdown('<div class="section-header">Generation Controls</div>', unsafe_allow_html=True)

        checkpoints = get_available_checkpoints()
        if checkpoints:
            selected_ckpt = st.selectbox("Checkpoint", checkpoints, index=len(checkpoints)-1)
            ckpt_path = os.path.join(CHECKPOINT_DIR, selected_ckpt)

            n_images = st.slider("Number of images", 4, 64, 16, step=4)
            nrow = st.slider("Images per row", 2, 8, 4)
            seed = st.number_input("Random seed (-1 = random)", value=-1, step=1)
            interpolate = st.checkbox("🔀 Latent space interpolation", value=False)

            st.markdown("")
            generate_btn = st.button("⚡ Generate Images", use_container_width=True)

            if generate_btn:
                with st.spinner("Generating..."):
                    G, ckpt_data = load_generator(ckpt_path)
                    if seed != -1:
                        torch.manual_seed(int(seed))
                    with torch.no_grad():
                        z = torch.randn(n_images, LATENT_DIM, device=device)
                        if interpolate and n_images >= 2:
                            z1 = torch.randn(1, LATENT_DIM, device=device)
                            z2 = torch.randn(1, LATENT_DIM, device=device)
                            alphas = torch.linspace(0, 1, n_images, device=device).unsqueeze(1)
                            z = z1 * (1 - alphas) + z2 * alphas
                        imgs = G(z)
                    pil_img = tensor_grid_to_pil(imgs, nrow=nrow)
                    st.session_state["last_gen"] = pil_img
                    st.session_state["last_imgs"] = imgs

            if "last_gen" in st.session_state:
                buf = io.BytesIO()
                st.session_state["last_gen"].save(buf, format="PNG")
                st.download_button(
                    "⬇️  Download Grid",
                    data=buf.getvalue(),
                    file_name="wgan_generated.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.info("No checkpoints found. Please train the model first.")

    with col_out:
        st.markdown('<div class="section-header">Generated Output</div>', unsafe_allow_html=True)
        if "last_gen" in st.session_state:
            st.image(st.session_state["last_gen"], use_column_width=True,
                     caption=f"WGAN-generated CIFAR-10-style images")
        else:
            st.markdown("""
            <div style="background:#0d1a26; border:1px dashed #2a3f52; border-radius:12px;
                        height:300px; display:flex; align-items:center; justify-content:center;
                        color:#4a6070; font-family:'Space Mono',monospace; font-size:0.9rem;">
                Generate images using the controls on the left
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════ Tab 3: Training Progress ═════════════════════════ #
with tab3:
    st.markdown('<div class="section-header">Training Progress</div>', unsafe_allow_html=True)

    checkpoints = get_available_checkpoints()
    if checkpoints:
        ckpt_path_prog = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        ckpt_data = torch.load(ckpt_path_prog, map_location="cpu")

        g_losses = ckpt_data.get("g_losses", [])
        c_losses = ckpt_data.get("c_losses", [])
        epoch_reached = ckpt_data.get("epoch", 0) + 1

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{epoch_reached}</div>
              <div class="metric-label">Epochs Trained</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            last_g = g_losses[-1] if g_losses else 0
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{last_g:.3f}</div>
              <div class="metric-label">Last G Loss</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            last_c = c_losses[-1] if c_losses else 0
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{last_c:.3f}</div>
              <div class="metric-label">Last Critic Loss</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        if g_losses and c_losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                            facecolor="#0d1a26")
            for ax in [ax1, ax2]:
                ax.set_facecolor("#0d1a26")
                ax.tick_params(colors="#8ca0b0")
                ax.spines[:].set_color("#2a3f52")

            ax1.plot(c_losses, color="#64c8ff", linewidth=1.5)
            ax1.set_title("Critic Loss", color="#e8f4f8")

            ax2.plot(g_losses, color="#ff9f64", linewidth=1.5)
            ax2.set_title("Generator Loss", color="#e8f4f8")

            plt.tight_layout()
            st.pyplot(fig)

        sample_files = get_sample_images()
        if sample_files:
            st.markdown('<div class="section-header">Sample Progression</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(sample_files), 6))
            for i, sf in enumerate(sample_files[-6:]):
                with cols[i]:
                    img = Image.open(os.path.join(SAMPLE_DIR, sf))
                    st.image(img, use_column_width=True)
    else:
        st.info("No training data found yet.")

    if st.button("🔄 Refresh Progress"):
        st.rerun()