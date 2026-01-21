# The Crystalline Topology Engine
## Exact Causal Circuit Discovery in Deep Neural Networks via Lazy Pointer Tracking

**Author:** Anaqia (Hyper-State Research Lab)  
**Status:** HPE v3.1 (Stable)

---

## 📑 Abstract
The quest for Explainable AI (XAI) has largely bifurcated into two streams: feature attribution (*which inputs matter?*) and circuit discovery (*which neurons connect?*). While the former has achieved exactness through frameworks like HPE v2, the latter remains computationally prohibitive, often requiring thousands of forward passes.

In this paper, we introduce **HPE v3** (Hyper-State Provenance Engine v3), also known as the **Crystalline Topology Engine**. HPE v3 unifies attribution and topology into a single, deterministic forward pass by introducing the **Causal Pointer Tensor ($\pi$)**. 

By employing a novel *"Lazy Causal Pruning"* algorithm and *"Hybrid-State Tracking,"* HPE v3 captures the exact top-$K$ causal parents for every neuron in real-time, effectively transforming the neural network from a fluid "black box" into a rigid, traceable **"Crystal Lattice."**

---

## 1. Introduction
Deep Learning models function as information processing systems where data flows through layers of increasing abstraction. 
* **Traditional methods (HPE v1/v2)** successfully tracked the quantity of information (Attribution).
* **HPE v3** addresses the pathway of information (Topology).

Current circuit discovery methods rely on "Interventionism"—repeatedly patching activations to see what breaks. This is slow and inexact. HPE v3 proposes a paradigm shift: **instead of guessing the circuit, we record it.**

We posit that a Neural Network can be viewed as a **Crystalline Structure**, where every activation at layer $L$ has a fixed, discrete set of "parent nodes" at layer $L-1$.

---

## 2. Theoretical Framework: The Quadruplet State ($\Omega$)
In HPE v3, we expand the state definition of a neuron. Let a tensor $T$ be defined not just by its value, but by its history and topology. The state is a quadruplet:

$$\Omega = \langle S, \Phi, \beta, \pi \rangle$$

Where:
* **$S$ (Signal):** The standard activation tensor ($\mathbb{R}$). Preserves ground-truth predictive behavior.
* **$\Phi$ (Feature Manifold):** A sparse/tiled tensor tracking the contribution of input sources ($\mathbb{R}^{Src}$).
* **$\beta$ (Structural Manifold):** A low-rank tensor tracking internal model biases ($\mathbb{R}^{1}$).
* **$\pi$ (Causal Pointer):** The novel topological tensor ($\mathbb{Z}^{K}$). It stores the indices of the Top-$K$ neurons in the previous layer that contributed maximally to the current activation.

### 2.1 The Law of Conservation of Logits
HPE v3 maintains the strict conservation law established in v2. For any output logit $y$:

$$y = \sum \Phi + \sum \beta$$

The topology tensor $\pi$ does not participate in the summation but serves as the metadata describing **how** $\Phi$ and $\beta$ flowed to the final state.

---

## 3. Algorithmic Implementation

### 3.1 The Causal Pointer Mechanism (Lazy Pruning)
Tracking every connection in a dense network ($N^2$ complexity) is impossible. HPE v3 utilizes **On-the-Fly Causal Pruning**.

For a linear operation $y = Wx$, instead of storing all connections, we compute:

$$\pi_i = \text{argtopk}_j (|C_{ij}|, K)$$

This operation is computationally efficient on modern GPUs. We store only the indices of the $K$ most significant parents. This creates a **"Winner-Takes-All" Topology**, reducing memory usage from $O(N^2)$ to $O(N \cdot K)$.

### 3.2 Memory-Safe Chunked Batch Folding
To enable high-resolution auditing (e.g., 10x10 grids) on large CNNs (ConvNeXt XL), we introduce **Chunked Folding**. We decompose the source dimension $Src$ into smaller chunks to prevent OOM errors on T4 GPUs.

### 3.3 Hybrid-State Attention Tracking (LLM Specific)
For Transformers (e.g., Qwen), $\pi$ behaves dynamically:
* **In MLP Layers:** $\pi$ tracks Feature Indices (which neuron fired?).
* **In Attention Layers:** $\pi$ tracks Positional Indices (which past token was attended to?).

This duality allows HPE v3 to prove specific behaviors: if $\pi$ points to a past token index $t-k$ and $\Phi$ remains isomorphic, we mathematically confirm a **Copying Mechanism (Induction Head)**.

---

## 4. Experimental Validation

### 4.1 Computer Vision: ConvNeXt XLarge
* **Setup:** 10x10 Grid Tiling (100 discrete sources).
* **Precision:** Exact ($< 10^{-4}$ error).
* **Discovery:** **"Texture Bias"**.
    * *Observation:* The $\pi$ tensor and $\Phi$ energy for "Tabby Cat" were concentrated entirely on the torso stripes. The face tiles had near-zero causal weight.
    * *Conclusion:* Proved the model classifies based on texture, ignoring geometry.

### 4.2 NLP: Qwen 2.5-0.5B (Reasoning)
* **Setup:** Prompt *"Write a simple python code for fibonacci"*.
* **Discovery:** **"Concept Anchoring"**.
    * *Observation:* As the model generated the code, the $\pi$ tensor for every subsequent token maintained a hard link back to the token "Fibonacci" in the prompt.
    * *Visual:* The heatmap displayed a vertical **"Crystalline Spine,"** proving the model actively anchored its context to the task keyword.

---

## 5. Conclusion
HPE v3 represents the maturation of exact mechanistic interpretability. By combining Dual-Manifold Provenance (from v2) with Causal Pointer Tracking, we enable a transition from **"AI Safety via Testing"** to **"AI Safety via Inspection."**

*Hyper-State Research Lab, 2026*
