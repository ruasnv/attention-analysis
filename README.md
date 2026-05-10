## Analysis of Attention as Geometric Operators

Spectral and geometric analysis of transformer attention weights as operators acting on the residual stream. This study evaluates the mechanistic interpretability of attention mechanisms and develops novel methods for structural weight analysis.

Some findings include:

* **MP Bulk Information:** Singular values within the Marchenko-Pastur bulk carry functional information in attention weights.


* **Indefinite Sign Structure:** The Query-Key circuit exhibits an indefinite eigenvalue structure, suggesting feature repulsion is a core component of routing.


* **Polar Component Roles:** Polar factors ($Q$ and $S$) play functionally distinct roles across different layers.


* **Spectral Variation:** Spectral geometry varies significantly layer-by-layer, tracking a phase transition from feature extraction to uniform routing.



---

## Experimental Infrastructure

* **Model:** GPT-2 (12 layers, 768 dimensions, 12 heads, 85M parameters).


* **Data:** WikiText-2 (benchmarking) and OpenWebText (fine-tuning).


* **Benchmarks:** LAMBADA (long-range context), HellaSwag, ARC, PIQA, and Winogrande (reasoning).


* **Tools:** SVD, Polar Decomposition, Marchenko-Pastur RMT, and Procrustes alignment.



---

## Complete Research Summary

### I. The Value ($W_V$) and Output ($W_O$) Circuits

* **Isometry of $W_V$:** Analysis reveals $W_V$ is 34–223$\times$ more orthogonal than random baselines, acting primarily as a rotation matrix.


* **Functional Polar Factors:** Q-surgery (rotation) preserves model function, while S-surgery (stretch) causes catastrophic degradation.


* **$W_O$ as an Amplifier:** Unlike $W_V$, $W_O$ is a trained directional amplifier with high MP outlier mass, directly correcting $W_V$’s imperfections to produce isometric per-head writes.


* **Alignment Phase Transition:** $W_V$ aligns with activation principal components in early layers (feature extraction) but inverts this alignment in late layers (uniform routing).



### II. The Query-Key ($G$) Routing Circuit

* **Geometric Asymmetry:** $W_Q$ and $W_K$ span nearly orthogonal subspaces (mean angles 60°–72°), making attention routing fundamentally directional.


* **Indefinite Bilinear Forms:** The $G$ operator ($W_Q W_K^T$) is structurally indefinite, encoding both attraction (positive eigenvalues) and repulsion (negative eigenvalues).


* **PMI Correlation:** The spectral magnitude of $G$ has a 0.945 rank correlation with Pointwise Mutual Information (PMI), though the specific sign structure is learned via training.



### III. Functional Validation via Weight Surgery

* **Kernel PCA Validation:** Replacing $W_V$ with a Procrustes-aligned PCA matrix improves long-range performance (LAMBADA), validating the KPCA framework.


* **Orientation Cost:** Surgery using raw PCA without Procrustes alignment leads to total model collapse, proving the residual stream's coordinate-sensitive nature.


* **Amplification Gap:** While PMI predicts the routing structure, it requires a ~6$\times$ amplification factor to overcome the Softmax bottleneck.



### IV. Compression & Scaling Laws

* **Rank Budget:** Every attention head utilizes its full rank-64 budget; no heads are naturally degenerate.


* **The Squareness Theorem:** For square matrices, MP-guided pruning is mathematically identical to naive SVD, limiting the utility of standard RMT pruning in attention layers.


* **Global vs. Single-Layer Surgery:** Single-layer interventions are highly resilient, but global geometric changes require fine-tuning to realign inter-layer calibration.