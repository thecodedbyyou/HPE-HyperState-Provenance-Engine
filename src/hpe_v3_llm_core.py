# ==============================================================================
# HYPER-STATE PROVENANCE ENGINE (HPE) v3.1 - CRYSTALLINE TOPOLOGY
# Target: Qwen 2.5-0.5B-Instruct
# Novelty: Causal Lattice Tracking with Memory-Safe Chunking (T4 Optimized)
# ==============================================================================

import subprocess
import sys
import gc
import math
import warnings
import os

# 1. SETUP & DEPENDENCIES
# ------------------------------------------------------------------------------
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

print("⚙️ Initializing HPE v3.1 Environment...")
try:
    import transformers
    import accelerate
    import tiktoken
except ImportError:
    print("📦 Installing dependencies (transformers, accelerate, tiktoken)...")
    install_package("transformers")
    install_package("accelerate")
    install_package("tiktoken")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# System Configuration
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMETERS ---
TOP_K_POINTERS = 4       # Jumlah "Parents" yang disimpan di tensor Pi
CHUNK_SIZE = 1024        # Ukuran batch komputasi untuk menghindari OOM pada LM Head

print(f"🚀 HPE v3.1 CRYSTAL ENGINE ONLINE: {device}")
print(f"💎 Tracking Depth: Top-{TOP_K_POINTERS} Causal Links per Neuron")
print(f"🛡️ Memory Safety: Chunked Processing (Size: {CHUNK_SIZE})")
print("🧠 TARGET: Qwen/Qwen2.5-0.5B-Instruct")

# ==============================================================================
# 2. CORE ENGINE: THE QUADRUPLET STATE (Omega)
# ==============================================================================

class HoloState_v3:
    """
    State Container v3 (The Crystalline Structure).
    
    Signal (S) : [B, Seq, Hidden] -> Ground Truth
    Phi (Φ)    : [B, Seq, Hidden, Source_Seq] -> Input Feature Provenance
    Beta (β)   : [B, Seq, Hidden, 1] -> Structural Bias
    Pi (π)     : [B, Seq, Hidden, TopK] -> Causal Pointer (Int16)
    """
    def __init__(self, signal, phi, beta, pi, pointer_type="feature"):
        self.signal = signal
        self.phi = phi
        self.beta = beta
        self.pi = pi
        self.pointer_type = pointer_type # 'feature' (MLP) or 'position' (Attention)

    @staticmethod
    def from_embedding(embed_layer, input_ids):
        # 1. Standard Signal
        signal = embed_layer(input_ids) # [B, S, H]
        B, Seq, Hidden = signal.shape
        
        # 2. Beta (Structure) -> Start at Zero
        beta = torch.zeros(B, Seq, Hidden, 1, device=signal.device, dtype=signal.dtype)
        
        # 3. Phi (Feature) -> Identity Matrix (Diagonal Scatter)
        # Token i bertanggung jawab atas Token i
        phi = torch.zeros(B, Seq, Hidden, Seq, device=signal.device, dtype=signal.dtype)
        
        # Diagonal Scatter Loop (Exact)
        for i in range(Seq):
            phi[:, i, :, i] = signal[:, i, :]
            
        # 4. Pi (Pointer) -> Init to Self Index
        # [B, S, H, K] -> Int16
        pi = torch.arange(Hidden, device=signal.device, dtype=torch.int16)
        pi = pi.view(1, 1, Hidden, 1).expand(B, Seq, Hidden, TOP_K_POINTERS)
        
        return HoloState_v3(signal, phi, beta, pi, pointer_type="feature")

    def clone(self):
        return HoloState_v3(
            self.signal.clone(), 
            self.phi.clone(), 
            self.beta.clone(), 
            self.pi.clone(),
            self.pointer_type
        )

# ==============================================================================
# 3. ATOMIC OPERATIONS (MEMORY SAFE CHUNKING)
# ==============================================================================

class HoloRMSNorm_v3(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, inp: HoloState_v3):
        x = inp.signal.float()
        
        # RMS Calculation
        variance = x.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.eps)
        
        # Affine Decomposition
        scale = self.weight.float() * rsqrt
        scale = scale.to(inp.signal.dtype)
        scale_broad = scale.unsqueeze(-1)
        
        # Apply
        out_signal = inp.signal * scale
        out_beta = inp.beta * scale_broad
        out_phi = inp.phi * scale_broad
        
        # Topology Handling: Pointer tidak berubah saat normalisasi (hanya scaling)
        out_pi = inp.pi.clone()
        
        return HoloState_v3(out_signal, out_phi, out_beta, out_pi, inp.pointer_type)

class HoloLinear_v3(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, inp: HoloState_v3):
        # 1. Signal (Standard GPU Op)
        out_signal = self.linear(inp.signal)
        
        # 2. Beta (Bias Propagation)
        beta_in = inp.beta.squeeze(-1)
        out_beta = self.linear(beta_in).unsqueeze(-1)
        
        # 3. Phi (Feature Propagation - Weight Only)
        original_bias = self.linear.bias
        self.linear.bias = None 
        
        phi_in = inp.phi.transpose(-1, -2)
        out_phi = self.linear(phi_in) 
        out_phi = out_phi.transpose(-1, -2)
        
        self.linear.bias = original_bias 
        
        # 4. Pi (THE CRYSTAL LATTICE GENERATION - MEMORY SAFE FIX)
        # Masalah: [B, S, Out, In] terlalu besar jika Out=152k.
        # Solusi: Iterasi (Chunking) pada dimensi Output.
        
        # Input: x [B, S, In]
        # Weight: w [Out, In]
        
        x_in = inp.signal # [B, S, In]
        batch_size, seq_len, _ = x_in.shape
        
        pi_list = []
        
        with torch.no_grad():
            x_expanded = x_in.unsqueeze(2) # [B, S, 1, In]
            
            # Loop per chunk of output features
            for i in range(0, self.out_features, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, self.out_features)
                
                # Slice weight: [Chunk, In]
                w_chunk = self.linear.weight[i:end_idx, :]
                
                # Reshape weight: [1, 1, Chunk, In]
                w_chunk_expanded = w_chunk.view(1, 1, end_idx - i, self.in_features)
                
                # Calculate Contribution [B, S, Chunk, In]
                # Ini aman karena Chunk kecil (1024)
                contrib_chunk = x_expanded * w_chunk_expanded
                
                # Ambil Top-K Indices pada dimensi INPUT (Last Dim)
                # Artinya: Neuron input mana yang paling memicu neuron output di chunk ini?
                _, indices = torch.topk(torch.abs(contrib_chunk), k=TOP_K_POINTERS, dim=-1) # [B, S, Chunk, K]
                
                # Simpan sebagai Int16 untuk hemat memori
                pi_list.append(indices.to(torch.int16))
                
                # Force cleanup intermediate tensor
                del contrib_chunk
                del w_chunk_expanded
            
            # Gabungkan kembali chunks menjadi [B, S, Out, K]
            out_pi = torch.cat(pi_list, dim=2)
            
        return HoloState_v3(out_signal, out_phi, out_beta, out_pi, pointer_type="feature")

class HoloSwiGLU_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = HoloLinear_v3(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = HoloLinear_v3(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = HoloLinear_v3(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: HoloState_v3):
        # 1. Projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # 2. Activation
        gate_act = self.act_fn(gate.signal)
        gate_scale = gate_act.unsqueeze(-1)
        
        # Inter State
        inter_signal = up.signal * gate_act
        inter_beta = up.beta * gate_scale
        inter_phi = up.phi * gate_scale
        
        # Pi Logic: Ikuti path 'UP' (Konten), Gate hanya modulator.
        inter_pi = up.pi
        
        inter_state = HoloState_v3(inter_signal, inter_phi, inter_beta, inter_pi, pointer_type="feature")
        
        # 3. Down Project
        return self.down_proj(inter_state)

class HoloRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class HoloAttention_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = HoloLinear_v3(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = HoloLinear_v3(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = HoloLinear_v3(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = HoloLinear_v3(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary = HoloRotaryEmbedding(self.head_dim)

    def forward(self, x: HoloState_v3):
        B, Seq, _ = x.signal.shape
        
        # 1. Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. RoPE
        q_sig = q.signal.view(B, Seq, self.num_heads, self.head_dim).transpose(1, 2)
        k_sig = k.signal.view(B, Seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_sig = v.signal.view(B, Seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(Seq)
        q_sig, k_sig = apply_rotary_pos_emb(q_sig, k_sig, cos, sin)
        
        if self.num_groups > 1:
            k_sig = k_sig.repeat_interleave(self.num_groups, dim=1)
            v_sig = v_sig.repeat_interleave(self.num_groups, dim=1)
            
        # 3. Attention Weights
        attn_scores = torch.matmul(q_sig, k_sig.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(Seq, Seq, device=device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_sig.dtype)
        
        # HPE v3: ATTENTION POINTER (Inter-Token Causality)
        with torch.no_grad():
            # Mean across heads for global topology
            attn_avg = attn_weights.mean(dim=1) # [B, SeqQ, SeqK]
            _, attn_indices = torch.topk(attn_avg, k=TOP_K_POINTERS, dim=-1) # [B, SeqQ, K]
            attn_pi = attn_indices.to(torch.int16) 
            # Expand to match [B, S, Hidden, K] format
            attn_pi_expanded = attn_pi.unsqueeze(2).expand(B, Seq, self.hidden_size, TOP_K_POINTERS)

        # 4. Manifold Propagation
        def split_heads(t, heads):
            last = t.shape[-1]
            return t.view(B, Seq, heads, self.head_dim, last).permute(0, 2, 1, 3, 4)

        v_phi = split_heads(v.phi, self.num_key_value_heads)
        v_beta = split_heads(v.beta, self.num_key_value_heads)
        
        if self.num_groups > 1:
            v_phi = v_phi.repeat_interleave(self.num_groups, dim=1)
            v_beta = v_beta.repeat_interleave(self.num_groups, dim=1)

        out_signal = torch.matmul(attn_weights, v_sig)
        out_phi = torch.einsum('bhqk,bhkds->bhqds', attn_weights, v_phi)
        out_beta = torch.einsum('bhqk,bhkds->bhqds', attn_weights, v_beta)
        
        # 5. Merge Heads
        def merge_heads(t):
            return t.permute(0, 2, 1, 3, 4).contiguous().reshape(B, Seq, self.hidden_size, -1)
            
        out_signal = out_signal.transpose(1, 2).contiguous().reshape(B, Seq, self.hidden_size)
        out_phi = merge_heads(out_phi)
        out_beta = merge_heads(out_beta)
        
        # Switch pointer type to 'position'
        curr_state = HoloState_v3(out_signal, out_phi, out_beta, attn_pi_expanded, pointer_type="position")
        
        return self.o_proj(curr_state)

# ==============================================================================
# 4. ARCHITECTURE: HOLO-QWEN v3
# ==============================================================================

class HoloQwenDecoderLayer_v3(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = HoloAttention_v3(config)
        self.mlp = HoloSwiGLU_v3(config)
        self.input_layernorm = HoloRMSNorm_v3(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HoloRMSNorm_v3(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: HoloState_v3):
        # Attention Block
        residual = x.clone()
        x_norm = self.input_layernorm(x)
        x_attn = self.self_attn(x_norm) 
        
        x_attn.signal = x_attn.signal + residual.signal
        x_attn.phi = x_attn.phi + residual.phi
        x_attn.beta = x_attn.beta + residual.beta
        
        # MLP Block
        residual = x_attn.clone()
        x_norm2 = self.post_attention_layernorm(x_attn)
        x_mlp = self.mlp(x_norm2) 
        
        x_mlp.signal = x_mlp.signal + residual.signal
        x_mlp.phi = x_mlp.phi + residual.phi
        x_mlp.beta = x_mlp.beta + residual.beta
        
        return x_mlp

class HoloQwen2_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HoloQwenDecoderLayer_v3(config, i) for i in range(config.num_hidden_layers)])
        self.norm = HoloRMSNorm_v3(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = HoloLinear_v3(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        # 1. Embed
        x = HoloState_v3.from_embedding(self.embed_tokens, input_ids)
        
        # 2. Layers
        layer_pointers = []
        for layer in self.layers:
            x = layer(x)
            # Simpan pointer (bisa dari MLP atau Attention, disini kita ambil snapshot state)
            layer_pointers.append(x.pi.detach().cpu())
            
        # 3. Final Norm
        x = self.norm(x)
        
        # 4. Head
        # Ini akan memanggil HoloLinear_v3 yang sudah di-patch dengan Chunking
        return self.lm_head(x), layer_pointers

# ==============================================================================
# 5. TRANSPLANTATION SURGERY
# ==============================================================================

def transplant_qwen_v3(holo_model, hf_model):
    print(">>> 🩺 Starting Brain Transplant (Weights Transfer)...")
    
    holo_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data
    
    for i, h_layer in enumerate(holo_model.layers):
        hf_layer = hf_model.model.layers[i]
        
        h_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data
        h_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data
        
        h_layer.self_attn.q_proj.linear.weight.data = hf_layer.self_attn.q_proj.weight.data
        h_layer.self_attn.q_proj.linear.bias.data = hf_layer.self_attn.q_proj.bias.data
        h_layer.self_attn.k_proj.linear.weight.data = hf_layer.self_attn.k_proj.weight.data
        h_layer.self_attn.k_proj.linear.bias.data = hf_layer.self_attn.k_proj.bias.data
        h_layer.self_attn.v_proj.linear.weight.data = hf_layer.self_attn.v_proj.weight.data
        h_layer.self_attn.v_proj.linear.bias.data = hf_layer.self_attn.v_proj.bias.data
        h_layer.self_attn.o_proj.linear.weight.data = hf_layer.self_attn.o_proj.weight.data
        
        h_layer.mlp.gate_proj.linear.weight.data = hf_layer.mlp.gate_proj.weight.data
        h_layer.mlp.up_proj.linear.weight.data = hf_layer.mlp.up_proj.weight.data
        h_layer.mlp.down_proj.linear.weight.data = hf_layer.mlp.down_proj.weight.data

    holo_model.norm.weight.data = hf_model.model.norm.weight.data
    holo_model.lm_head.linear.weight.data = hf_model.lm_head.weight.data
    print("✅ Transplant Complete. The Crystal Engine is Alive.")

# ==============================================================================
# 6. FORENSIC AUDIT & VISUALIZATION LOGIC
# ==============================================================================

def hpe_audit_v3(prompt_text):
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n>>> Loading Host Model: {model_id}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    holo_model = HoloQwen2_v3(hf_model.config).to(device)
    transplant_qwen_v3(holo_model, hf_model)
    holo_model.eval()
    
    messages = [
        {"role": "system", "content": "You are a helpful python assistant."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    
    print(f"\n📝 PROMPT: {prompt_text}")
    print("⏳ Generating Response (Autoregressive with Topology Tracking)...")
    
    generated_ids = input_ids.clone()
    max_new_tokens = 20 # Tetap pendek agar visualisasi enak dilihat
    
    final_snapshot = None
    final_token_id = None
    
    # Generation Loop
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out_head, pointers = holo_model(generated_ids)
            
            next_token_logits = out_head.signal[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            
            # Save snapshot
            if _ == max_new_tokens - 1:
                final_snapshot = out_head
                final_token_id = next_token_id
            
            if next_token_id == tokenizer.eos_token_id:
                # Jika EOS sebelum max token, simpan snapshot ini juga
                final_snapshot = out_head
                final_token_id = next_token_id
                break
                
    generated_text = tokenizer.decode(generated_ids[0])
    print(f"\n🤖 QWEN GENERATED:\n{generated_text}")
    
    # --- AUDIT PHASE 1: EXACTNESS ---
    out = final_snapshot
    target_idx = final_token_id
    
    # Pindahkan ke CPU untuk kalkulasi
    phi_contribs = out.phi[0, -1, target_idx, :].cpu()
    beta_contrib = out.beta[0, -1, target_idx, 0].item()
    phi_total = phi_contribs.sum().item()
    
    logit_recon = phi_total + beta_contrib
    logit_actual = out.signal[0, -1, target_idx].item()
    
    print("\n📊 [AUDIT REPORT 1: EXACTNESS]")
    print(f"Logit Actual    : {logit_actual:.8f}")
    print(f"HPE Reconstruct : {logit_recon:.8f}")
    diff = abs(logit_actual - logit_recon)
    if diff < 1e-4:
        print(f"✅ STATUS: SECURE (Bit-Perfect). Diff: {diff:.9f}")
    else:
        print(f"❌ STATUS: LEAKAGE. Diff: {diff:.9f}")
        
    # --- AUDIT PHASE 2: TOPOLOGY ---
    # last_pi: [K] indices
    last_pi = out.pi[0, -1, target_idx, :].cpu().numpy()
    
    print("\n💎 [AUDIT REPORT 2: CRYSTALLINE TOPOLOGY]")
    print(f"Target Token: '{tokenizer.decode([target_idx])}'")
    print(f"Triggered by Hidden Neurons (Last Layer): {last_pi}")
    
    # --- VISUALIZATION ---
    tokens_text = [tokenizer.decode([t]) for t in generated_ids[0]]
    vals = phi_contribs.numpy()
    
    # Handle mismatch length (jika phi lebih panjang/pendek dari tokens karena special tokens)
    # Phi source biasanya sama dengan input sequence length
    seq_len = len(tokens_text)
    vals = vals[:seq_len]
    
    # Top 10 Features
    indices = np.argsort(np.abs(vals))[-10:]
    top_tokens = [tokens_text[i] for i in indices]
    top_vals = [vals[i] for i in indices]
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Bar Chart
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_vals]
    ax[0].barh(top_tokens, top_vals, color=colors)
    ax[0].set_title(f"HPE v2: Semantic Attribution for '{tokenizer.decode([target_idx])}'")
    ax[0].set_xlabel("Logit Impact")
    ax[0].grid(axis='x', alpha=0.3)
    
    # 2. Crystal Lattice Heatmap
    # Mengambil Phi Matrix untuk melihat koneksi Token-to-Token
    # out.phi shape: [1, Seq, Hidden, Source]
    # Kita normalkan dimensi Hidden -> [Seq, Source]
    phi_matrix = out.phi[0, :, :, :].norm(dim=1).cpu().detach().numpy()
    
    # Potong sesuai panjang sequence yang ada
    phi_matrix = phi_matrix[:seq_len, :seq_len]
    
    # Masking upper triangle (causal mask) agar visualisasi bersih
    mask = np.triu(np.ones_like(phi_matrix, dtype=bool), k=1)
    
    sns.heatmap(phi_matrix, ax=ax[1], cmap="magma", mask=mask,
                xticklabels=tokens_text, yticklabels=tokens_text)
    ax[1].set_title("HPE v3: Crystalline Lattice (Token Connectivity Map)")
    ax[1].set_xlabel("Source Token (Cause)")
    ax[1].set_ylabel("Generated Token (Effect)")
    
    plt.tight_layout()
    plt.show()

# RUN
task_prompt = "Write a simple python code for fibonacci sequence."
hpe_audit_v3(task_prompt)

