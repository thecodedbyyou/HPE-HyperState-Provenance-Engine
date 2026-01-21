# ==============================================================================
# HYPER-STATE PROVENANCE ENGINE (HPE) v3.1 - CRYSTAL VISION (FINAL FIXED)
# Target: ConvNeXt XLarge (10x10 Tiled Crystalline Topology)
# Fix: Tiling Boundary Leakage & Exact Decomposition Logic
# ==============================================================================

import subprocess
import sys
import gc
import os
import math

# 1. SETUP LINGKUNGAN
# ------------------------------------------------------------------------------
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

print("⚙️ Initializing HPE v3.1 Vision Environment...")
try:
    import timm
except ImportError:
    print("📦 Installing timm...")
    install_package("timm")
    import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import warnings
import scipy.ndimage

# Konfigurasi Sistem
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KONFIGURASI TILING 10x10 (Resolusi Tinggi)
GRID_SIZE = 10 
NUM_SOURCES = GRID_SIZE * GRID_SIZE # 100 Sumber
TOP_K_POINTERS = 3 
CONV_CHUNK_SIZE = 16 # Anti-OOM Batch Size

print(f"🚀 HPE v3.1 CRYSTAL VISION ONLINE: {device}")
print(f"🧠 TARGET: ConvNeXt XLarge (Tiled Mode: {GRID_SIZE}x{GRID_SIZE} = {NUM_SOURCES} Sources)")

# ==============================================================================
# 2. CORE ENGINE: THE QUADRUPLET STATE (Omega)
# ==============================================================================

class HoloState_v3:
    """
    State Container v3.
    Signal (S) : [B, C, H, W]
    Phi (Φ)    : [B, C, H, W, NUM_SOURCES] (Feature Manifold)
    Beta (β)   : [B, C, H, W, 1] (Structural Manifold)
    Pi (π)     : [B, C, H, W, TOP_K] (Causal Pointer - Int16)
    """
    def __init__(self, signal, phi, beta, pi):
        self.signal = signal
        self.phi = phi
        self.beta = beta
        self.pi = pi

    @staticmethod
    def from_input(img_tensor):
        # img_tensor: [B, 3, 224, 224]
        signal = img_tensor
        B, C, H, W = img_tensor.shape
        
        # 1. Beta (Structure) -> Init Zero
        beta = torch.zeros(*img_tensor.shape, 1, device=img_tensor.device)
        
        # 2. Phi (Feature) -> Tiled Mapping
        phi = torch.zeros(B, C, H, W, NUM_SOURCES, device=img_tensor.device)
        
        patch_h = H // GRID_SIZE
        patch_w = W // GRID_SIZE
        
        # 3. Pi (Pointer) -> Init dengan Source ID diri sendiri
        pi = torch.zeros(B, C, H, W, TOP_K_POINTERS, dtype=torch.int16, device=img_tensor.device)
        
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                src_id = r * GRID_SIZE + c
                
                # FIX LEAKAGE: Handle sisa pixel di pinggir (Boundary Check)
                h_start = r * patch_h
                h_end = (r + 1) * patch_h if r < GRID_SIZE - 1 else H
                
                w_start = c * patch_w
                w_end = (c + 1) * patch_w if c < GRID_SIZE - 1 else W
                
                # Fill Phi
                phi[:, :, h_start:h_end, w_start:w_end, src_id] = \
                    signal[:, :, h_start:h_end, w_start:w_end]
                
                # Fill Pi
                pi[:, :, h_start:h_end, w_start:w_end, :] = src_id
                    
        return HoloState_v3(signal, phi, beta, pi)

# ==============================================================================
# 3. ATOMIC OPERATIONS (CORRECTED DECOMPOSITION)
# ==============================================================================

class HoloConv2d_v3(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=bias)
        
    def forward(self, inp: HoloState_v3):
        # 1. SIGNAL PATH
        out_signal = self.conv(inp.signal)
        B, Cout, H_out, W_out = out_signal.shape
        
        # 2. BETA PATH
        # Apply Conv to Beta (accumulate bias from prev layers * weights + current bias)
        beta_in = inp.beta.squeeze(-1) # [B, Cin, H, W]
        out_beta = self.conv(beta_in).unsqueeze(-1) # [B, Cout, H, W, 1]
        
        # 3. PHI PATH
        # Disable bias for Phi to ensure pure feature propagation
        original_bias = self.conv.bias
        self.conv.bias = None
        
        Src = inp.phi.shape[-1]
        conv_out_phi = torch.zeros(B, Src, Cout, H_out, W_out, device=inp.signal.device)
        
        # Fold: [B, C, H, W, Src] -> [B, Src, C, H, W] -> [B*Src, C, H, W]
        phi_folded = inp.phi.permute(0, 4, 1, 2, 3).reshape(B*Src, *inp.phi.shape[1:4])
        
        # Chunked Convolution
        for i in range(0, B*Src, CONV_CHUNK_SIZE):
            end_idx = min(i + CONV_CHUNK_SIZE, B*Src)
            chunk = phi_folded[i:end_idx]
            res_chunk = self.conv(chunk)
            conv_out_phi[0, i:end_idx] = res_chunk
            del chunk, res_chunk
            
        # [B, Src, Cout, H, W] -> [B, Cout, H, W, Src]
        conv_out_phi = conv_out_phi.permute(0, 2, 3, 4, 1)
        
        # 4. PI PATH
        with torch.no_grad():
            # Find dominant source tiles
            _, indices = torch.topk(torch.abs(conv_out_phi), k=TOP_K_POINTERS, dim=-1)
            out_pi = indices.to(torch.int16)
        
        self.conv.bias = original_bias # Restore bias
        return HoloState_v3(out_signal, conv_out_phi, out_beta, out_pi)

class HoloLinear_v3(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        
    def forward(self, inp: HoloState_v3):
        # 1. Signal
        out_signal = self.linear(inp.signal)
        
        # 2. Beta
        beta_in = inp.beta.squeeze(-1)
        out_beta = self.linear(beta_in).unsqueeze(-1)
        
        # 3. Phi
        original_bias = self.linear.bias
        self.linear.bias = None
        
        # [..., In, Src] -> [..., Src, In]
        x = inp.phi.transpose(-1, -2)
        out_phi = self.linear(x).transpose(-1, -2)
        
        # 4. Pi
        with torch.no_grad():
            _, indices = torch.topk(torch.abs(out_phi), k=TOP_K_POINTERS, dim=-1)
            out_pi = indices.to(torch.int16)
        
        self.linear.bias = original_bias
        return HoloState_v3(out_signal, out_phi, out_beta, out_pi)

class HoloLayerNorm_v3(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
        self.data_format = data_format
        self.eps = eps
        
    def forward(self, inp: HoloState_v3):
        x = inp.signal
        
        # Manual LN Calculation to decompose Scale & Shift
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True, unbiased=False)
            
            # [C, 1, 1] for ConvNeXt
            weight = self.ln.weight[:, None, None]
            bias = self.ln.bias[:, None, None]
            
            scale = weight / torch.sqrt(var + self.eps)
            shift = bias - (u * scale)
            
            out_signal = (x - u) / torch.sqrt(var + self.eps) * weight + bias
            
            # Broadcast shapes for Beta/Phi
            # inp.beta: [B, C, H, W, 1]
            # scale: [B, C, H, W] -> [B, C, H, W, 1]
            scale_broad = scale.unsqueeze(-1)
            shift_broad = shift.unsqueeze(-1)
            
        else: # channels_last
            u = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            
            scale = self.ln.weight / torch.sqrt(var + self.eps)
            shift = self.ln.bias - (u * scale)
            
            out_signal = self.ln(x)
            scale_broad = scale.unsqueeze(-1)
            shift_broad = shift.unsqueeze(-1)
            
        out_beta = inp.beta * scale_broad + shift_broad
        out_phi = inp.phi * scale_broad
        out_pi = inp.pi.clone()
            
        return HoloState_v3(out_signal, out_phi, out_beta, out_pi)

class HoloGELU_v3(nn.Module):
    def forward(self, inp: HoloState_v3):
        out_signal = F.gelu(inp.signal)
        
        # Ratio Method
        mask = torch.abs(inp.signal) > 1e-6
        ratio = torch.zeros_like(inp.signal)
        ratio[mask] = out_signal[mask] / inp.signal[mask]
        ratio_exp = ratio.unsqueeze(-1)
        
        out_beta = inp.beta * ratio_exp
        out_phi = inp.phi * ratio_exp
        out_pi = inp.pi.clone()
            
        return HoloState_v3(out_signal, out_phi, out_beta, out_pi)

class HoloGlobalAvgPool_v3(nn.Module):
    def forward(self, inp: HoloState_v3):
        # Avg Pool spatial H,W -> [-2, -1]
        out_signal = inp.signal.mean([-2, -1])
        
        # Beta & Phi dimensions: [B, C, H, W, ...]
        # Mean over H, W (indices 2, 3)
        out_beta = inp.beta.mean(dim=[2, 3]) 
        out_phi = inp.phi.mean(dim=[2, 3])
        
        # Pi Logic: Recalculate dominant source for the pooled vector
        with torch.no_grad():
            _, indices = torch.topk(torch.abs(out_phi), k=TOP_K_POINTERS, dim=-1)
            out_pi = indices.to(torch.int16)
            
        return HoloState_v3(out_signal, out_phi, out_beta, out_pi)

# ==============================================================================
# 4. ARCHITECTURE: HOLO-CONVNEXT XL
# ==============================================================================

class HoloConvNeXtBlock_v3(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = HoloConv2d_v3(dim, dim, k=7, p=3, groups=dim)
        self.norm = HoloLayerNorm_v3(dim, eps=1e-6)
        self.pwconv1 = HoloLinear_v3(dim, 4 * dim)
        self.act = HoloGELU_v3()
        self.pwconv2 = HoloLinear_v3(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
    def forward(self, x: HoloState_v3):
        shortcut = x
        x = self.dwconv(x)
        
        # Permute NCHW -> NHWC
        x.signal = x.signal.permute(0, 2, 3, 1)
        x.beta = x.beta.permute(0, 2, 3, 1, 4)
        x.phi = x.phi.permute(0, 2, 3, 1, 4)
        x.pi = x.pi.permute(0, 2, 3, 1, 4)
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer Scale (Gamma)
        # Gamma shape: [dim]. x shape [B, H, W, C, ...]
        # Gamma broadcast: [dim] -> [1, 1, 1, dim, 1]
        gamma_broad = self.gamma.view(1, 1, 1, -1, 1)
        
        x.signal = x.signal * self.gamma # Auto-broadcast for signal
        x.beta = x.beta * gamma_broad
        x.phi = x.phi * gamma_broad
        
        # Permute Back
        x.signal = x.signal.permute(0, 3, 1, 2)
        x.beta = x.beta.permute(0, 3, 1, 2, 4)
        x.phi = x.phi.permute(0, 3, 1, 2, 4)
        x.pi = x.pi.permute(0, 3, 1, 2, 4)
        
        # Residual
        x.signal = x.signal + shortcut.signal
        x.beta = x.beta + shortcut.beta
        x.phi = x.phi + shortcut.phi
        
        return x

class HoloConvNeXt_v3(nn.Module):
    def __init__(self, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], num_classes=1000):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        
        stem = nn.Sequential(
            HoloConv2d_v3(3, dims[0], k=4, s=4),
            HoloLayerNorm_v3(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            ds = nn.Sequential(
                HoloLayerNorm_v3(dims[i], eps=1e-6, data_format="channels_first"),
                HoloConv2d_v3(dims[i], dims[i+1], k=2, s=2)
            )
            self.downsample_layers.append(ds)
            
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[HoloConvNeXtBlock_v3(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)
            
        self.norm = HoloLayerNorm_v3(dims[-1], eps=1e-6)
        self.head = HoloLinear_v3(dims[-1], num_classes)
        self.global_pool = HoloGlobalAvgPool_v3()

    def forward(self, x):
        if not isinstance(x, HoloState_v3): x = HoloState_v3.from_input(x)
        
        for i in range(4):
            print(f"  > Processing Stage {i}...")
            for layer in self.downsample_layers[i]: x = layer(x)
            for block in self.stages[i]: x = block(x)
            gc.collect() 
                
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.head(x)
        return x

# ==============================================================================
# 5. TRANSPLANT TOOLS
# ==============================================================================

def transplant_convnext_v3(holo_model, timm_model):
    print(">>> 🩺 Starting Brain Transplant (Weights Transfer)...")
    sd_timm = timm_model.state_dict()
    sd_holo = holo_model.state_dict()
    count = 0
    
    def copy(h, t):
        if t in sd_timm and h in sd_holo:
            if sd_holo[h].shape == sd_timm[t].shape:
                sd_holo[h].data.copy_(sd_timm[t].data)
                return 1
            elif sd_holo[h].numel() == sd_timm[t].numel():
                sd_holo[h].data.copy_(sd_timm[t].data.view(sd_holo[h].shape))
                return 1
        return 0

    count += copy("downsample_layers.0.0.conv.weight", "stem.0.weight")
    count += copy("downsample_layers.0.0.conv.bias", "stem.0.bias")
    count += copy("downsample_layers.0.1.ln.weight", "stem.1.weight")
    count += copy("downsample_layers.0.1.ln.bias", "stem.1.bias")
    
    for i in range(3):
        h_base = f"downsample_layers.{i+1}"
        t_base = f"stages.{i+1}.downsample"
        count += copy(f"{h_base}.0.ln.weight", f"{t_base}.0.weight")
        count += copy(f"{h_base}.0.ln.bias", f"{t_base}.0.bias")
        count += copy(f"{h_base}.1.conv.weight", f"{t_base}.1.weight")
        count += copy(f"{h_base}.1.conv.bias", f"{t_base}.1.bias")

    for i in range(4):
        for b in range(len(holo_model.stages[i])):
            h_blk = f"stages.{i}.{b}"
            t_blk = f"stages.{i}.blocks.{b}"
            count += copy(f"{h_blk}.dwconv.conv.weight", f"{t_blk}.conv_dw.weight")
            count += copy(f"{h_blk}.dwconv.conv.bias", f"{t_blk}.conv_dw.bias")
            count += copy(f"{h_blk}.norm.ln.weight", f"{t_blk}.norm.weight")
            count += copy(f"{h_blk}.norm.ln.bias", f"{t_blk}.norm.bias")
            count += copy(f"{h_blk}.pwconv1.linear.weight", f"{t_blk}.mlp.fc1.weight")
            count += copy(f"{h_blk}.pwconv1.linear.bias", f"{t_blk}.mlp.fc1.bias")
            count += copy(f"{h_blk}.pwconv2.linear.weight", f"{t_blk}.mlp.fc2.weight")
            count += copy(f"{h_blk}.pwconv2.linear.bias", f"{t_blk}.mlp.fc2.bias")
            count += copy(f"{h_blk}.gamma", f"{t_blk}.gamma")

    count += copy("norm.ln.weight", "head.norm.weight")
    count += copy("norm.ln.bias", "head.norm.bias")
    count += copy("head.linear.weight", "head.fc.weight")
    count += copy("head.linear.bias", "head.fc.bias")
    print(f"✅ Transplant Complete. Copied {count} tensors.")

# ==============================================================================
# 6. FORENSIC AUDIT & VISUALIZATION
# ==============================================================================

def hpe_audit_task(model, img_url, task_name, class_labels):
    print(f"\n{'='*60}")
    print(f"🏁 STARTING TASK: {task_name}")
    print(f"{'='*60}")
    
    response = requests.get(img_url)
    img_pil = Image.open(BytesIO(response.content)).convert('RGB')
    
    transform = timm.data.create_transform(
        input_size=(224, 224), is_training=False,
        mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD
    )
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    print(">>> Scanning Neural Topology...")
    with torch.no_grad():
        out = model(img_tensor)
    
    logits = out.signal[0]
    probs = torch.softmax(logits, dim=0)
    top3 = torch.topk(probs, 3)
    
    win_idx = top3.indices[0].item()
    win_label = class_labels[win_idx]
    win_logit = logits[win_idx].item()
    
    print(f"\n[MODEL PREDICTION]: {win_label} ({top3.values[0].item():.2%})")
    print(f"Logit Value: {win_logit:.4f}")

    # 3. Forensic Audit
    phi_total = out.phi[0, win_idx, :].sum().item()
    beta_total = out.beta[0, win_idx, 0].item()
    recon_logit = phi_total + beta_total
    
    print("\n[EXACTNESS CHECK]")
    print(f"HPE Reconstruction: {recon_logit:.6f}")
    diff = abs(win_logit - recon_logit)
    if diff < 1e-3:
        print(f"✅ SECURE. Diff: {diff:.8f}")
    else:
        print(f"❌ LEAKAGE. Diff: {diff:.8f}")
        
    # 4. HPE v3 Insight Generation
    phi_data = out.phi[0, win_idx, :].cpu()
    top_indices = torch.topk(phi_data.abs(), k=10).indices
    pi_data = out.pi[0, win_idx, :].cpu().numpy()
    
    print(f"\n💎 [CRYSTAL INSIGHT]")
    print(f"Dominant Tiles (Phi Energy) : {top_indices.numpy()}")
    print(f"Causal Pointers (Pi Link)   : {pi_data}")
    
    # 5. Visualization
    features = phi_data.view(GRID_SIZE, GRID_SIZE).numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # A. Original Image
    ax[0].imshow(img_pil.resize((224,224)))
    ax[0].set_title(f"Input: {task_name}")
    ax[0].axis('off')
    
    # B. Holographic Heatmap
    vmax = np.max(np.abs(features))
    features_smooth = scipy.ndimage.zoom(features, 224/GRID_SIZE, order=1)
    im = ax[1].imshow(features_smooth, cmap='seismic', vmin=-vmax, vmax=vmax, extent=[0, 224, 224, 0])
    ax[1].set_title(f"HPE v3.1 Feature Map\n(Red=Excite, Blue=Inhibit)")
    plt.colorbar(im, ax=ax[1])
    
    # C. Decision Anatomy
    bias_val = beta_total
    pos_feat = features[features > 0].sum()
    neg_feat = features[features < 0].sum()
    
    bars = ax[2].bar(['Global Bias', 'Pos Feat', 'Neg Feat'], 
                     [bias_val, pos_feat, neg_feat], 
                     color=['gray', 'red', 'blue'])
    ax[2].axhline(0, color='black', linewidth=0.8)
    ax[2].set_title("Decision Anatomy (Bias vs Features)")
    for bar in bars:
        yval = bar.get_height()
        ax[2].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom' if yval > 0 else 'top')

    plt.tight_layout()
    plt.show()
    
    # Separate Figure for Crystal Topology
    plt.figure(figsize=(6, 6))
    plt.imshow(img_pil.resize((224,224)), alpha=0.5)
    plt.title("Crystalline Topology (Dominant Tiles)")
    patch_size = 224 // GRID_SIZE
    
    for idx in top_indices:
        r, c = divmod(idx.item(), GRID_SIZE)
        rect = plt.Rectangle((c*patch_size, r*patch_size), patch_size, patch_size, 
                             linewidth=2, edgecolor='yellow', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(c*patch_size+5, r*patch_size+15, str(idx.item()), color='yellow', fontsize=8, fontweight='bold')
        
    plt.axis('off')
    plt.show()

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

# Init Model
print(">>> Init HPE v3.1 Architecture (ConvNeXt XL)...")
holo_model = HoloConvNeXt_v3(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]).to(device)
holo_model.eval()

# Load Weights
import time
print(">>> Downloading Pretrained Weights...")
for i in range(5):
    try:
        timm_model = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k', pretrained=True).to(device)
        timm_model.eval()
        break
    except:
        print("Retry download...")
        time.sleep(2)

transplant_convnext_v3(holo_model, timm_model)

# Labels
labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
class_labels = requests.get(labels_url).text.splitlines()[1:]

# TASK 1: DOG
hpe_audit_task(
    holo_model, 
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", 
    "Task 1: Pomeranian Dog", 
    class_labels
)

# TASK 2: CAT
hpe_audit_task(
    holo_model, 
    "http://images.cocodataset.org/val2017/000000039769.jpg", 
    "Task 2: Tabby Cat", 
    class_labels
)

print("\n>>> MISSION SUCCESS: HPE v3.1 Vision Audit Complete.")

