PAFT: Polar Adaptation Fine-Tuning
==================================

> **Surgical, Full-Rank, Zero-Latency Adaptation for Large Language Models**

PAFT is a novel Parameter-Efficient Fine-Tuning (PEFT) methodology that moves beyond the "additive noise" paradigm of methods like LoRA. Instead of adding low-rank matrices to pre-trained weights, PAFT utilizes **Polar Decomposition** to perform in-place geometric surgery on the model's existing attention manifold.

By isolating the **orthogonal routing manifold (Q)** and optimizing the **full-rank stretch tensor (S)**, PAFT achieves expert-level domain adaptation while mathematically preventing the "Directional Diversity Collapse" and inference latency common in modern adapters.

Key Advantages
--------------

*   **Inference Speed:** The model runs at the exact same inference speed as the original baseline.
    
*   **Geometric Stability:** By freezing the routing manifold (Q), PAFT protects foundational reasoning and core syntax, preventing catastrophic forgetting on benchmarks like Winogrande and HellaSwag.
    
*   **Spectral Health:** Maintains the **Spectral Entropy** and **Stable Rank** of the pre-trained model, ensuring feature diversity remains intact.
    

The Mathematics of PAFT
-----------------------

PAFT is built on the principle that any weight matrix W can be decomposed into two distinct geometric operations: W = QS

Where:

*   **Q (The rotation),** an orthogonal matrix representing pure rotation and reflection. This dictates where information moves (Routing).
    
*   **S (The Stretch):** A positive semi-definite matrix scaling the magnitude of specific feature vectors. This dictates what information is amplified (Magnitude).
    

### The Elastic Optimization

During training, we freeze Q to anchor the model's foundational grammar and optimize S as a general, unconstrained matrix. This allows the model to learn a **residual micro-rotation** (Q\_delta) alongside the feature magnitude updates:

W\_final = (Q\_frozen \* Q\_delta) S\_new

This elastic constraint allows the model to tilt its logic toward the target domain (e.g., Python or Legal text) without destroying its pre-trained base.

Architectural Integration: The OV Circuit
-----------------------------------------

PAFT targets the **Output-Value (OV) Circuit** of the Transformer attention block, leaving the Query-Key (QK) routing logic untouched:

*   Freeze Q\_v and train their importance S\_v.
    
*   Q\_o and train the signal volume S\_o.