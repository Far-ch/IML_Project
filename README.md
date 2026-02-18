# OOD Detection: From GMMs to Autoregressive Models (NADE)

This repository contains the implementation for the two-phase course project of **Introduction to Machine Learning** at **Sharif University of Technology**. The project focuses on Out-of-Distribution (OOD) detection, transitioning from classical density-based methods to modern neural autoregressive models.

---

## 📌 Project Overview
The goal of this project is to build systems that can distinguish between In-Distribution (ID) data (samples from classes seen during training) and Out-of-Distribution (OOD) data (unseen or "open-world" samples)

### Phase 1: Density-Based Detection with GMM
In this phase, the ID distribution is modeled using a **Gaussian Mixture Model (GMM)**
* **Methodology:** Explicitly model the ID data distribution $p_{ID}(x)$ as a weighted sum of Gaussian components.
* **Optimization:** Parameters are estimated using the **Expectation-Maximization (EM)** algorithm, implementing both E-steps and M-steps from scratch.
* **OOD Strategy:** A log-likelihood thresholding rule is applied: $\delta(x) = ID$ if $s(x) \ge \tau$, otherwise $OOD$.



### Phase 2: Autoregressive Modeling with NADE
This phase explores **Neural Autoregressive Distribution Estimation (NADE)** to capture complex, non-linear dependencies in high-dimensional image data.
* **Dataset:** Binarized MNIST, where digits 0-4 are treated as ID and digits 5-9 as OOD.
* **Core Architecture:** The joint probability is decomposed into a product of conditionals using the chain rule: $p(x) = \prod_{d=1}^{D} p(x_d | x_{<d})$.
* **Technical Highlights:** * Implementation of a weight-sharing computational "trick" to optimize the forward pass.
    * Training via minimization of Negative Log-Likelihood (NLL).
    * Evaluation of detection performance using **AUROC** metrics.
    * Sequential pixel-by-pixel image generation.



---

## 🛠️ Requirements
* Python 3.x
* PyTorch (for NADE architecture and tensor operations) 
* NumPy & SciPy (for EM implementation and mathematical operations) 
* Matplotlib (for loss plotting and OOD histogram visualization) 

---

## 🚀 Key Tasks Implemented
### Phase 1
- [x] Implementation of the EM algorithm (Initialization, E-step, M-step).
- [x] Likelihood thresholding for OOD detection.
- [x] Theoretical analysis of Mahalanobis distance and likelihood pitfalls in high dimensions.

### Phase 2
- [x] NADE class implementation with recursive hidden state updates.
- [x] Model training on MNIST digits 0-4.
- [x] OOD evaluation using digits 5-9 and AUROC calculation.
- [x] Image generation via sequential sampling.




