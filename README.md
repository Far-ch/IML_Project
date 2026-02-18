# OOD Detection: From GMMs to Autoregressive Models (NADE)

[cite_start]This repository contains the implementation for the two-phase course project of **Introduction to Machine Learning** at **Sharif University of Technology**. The project focuses on Out-of-Distribution (OOD) detection, transitioning from classical density-based methods to modern neural autoregressive models.

---

## 📌 Project Overview
[cite_start]The goal of this project is to build systems that can distinguish between In-Distribution (ID) data (samples from classes seen during training) and Out-of-Distribution (OOD) data (unseen or "open-world" samples)[cite: 35, 36, 37].

### Phase 1: Density-Based Detection with GMM
[cite_start]In this phase, the ID distribution is modeled using a **Gaussian Mixture Model (GMM)**[cite: 75, 91].
* [cite_start]**Methodology:** Explicitly model the ID data distribution $p_{ID}(x)$ as a weighted sum of Gaussian components[cite: 78, 91, 93].
* [cite_start]**Optimization:** Parameters are estimated using the **Expectation-Maximization (EM)** algorithm, implementing both E-steps and M-steps from scratch[cite: 117, 120].
* [cite_start]**OOD Strategy:** A log-likelihood thresholding rule is applied: $\delta(x) = ID$ if $s(x) \ge \tau$, otherwise $OOD$[cite: 85].



### Phase 2: Autoregressive Modeling with NADE
[cite_start]This phase explores **Neural Autoregressive Distribution Estimation (NADE)** to capture complex, non-linear dependencies in high-dimensional image data[cite: 191, 229, 230].
* [cite_start]**Dataset:** Binarized MNIST, where digits 0-4 are treated as ID and digits 5-9 as OOD[cite: 280, 282, 283].
* [cite_start]**Core Architecture:** The joint probability is decomposed into a product of conditionals using the chain rule: $p(x) = \prod_{d=1}^{D} p(x_d | x_{<d})$[cite: 231, 239].
* **Technical Highlights:** * Implementation of a weight-sharing computational "trick" to optimize the forward pass[cite: 243, 261].
    * Training via minimization of Negative Log-Likelihood (NLL)[cite: 267].
    * Evaluation of detection performance using **AUROC** metrics[cite: 301].
    * Sequential pixel-by-pixel image generation[cite: 289, 312].



---

## 🛠️ Requirements
* Python 3.x
* PyTorch (for NADE architecture and tensor operations) [cite: 23, 213]
* NumPy & SciPy (for EM implementation and mathematical operations) [cite: 23, 213]
* Matplotlib (for loss plotting and OOD histogram visualization) [cite: 294, 300]

---

## 🚀 Key Tasks Implemented
### Phase 1
- [x] Implementation of the EM algorithm (Initialization, E-step, M-step)[cite: 121, 124, 131].
- [x] Likelihood thresholding for OOD detection[cite: 145].
- [x] Theoretical analysis of Mahalanobis distance and likelihood pitfalls in high dimensions[cite: 161, 186].

### Phase 2
- [x] NADE class implementation with recursive hidden state updates[cite: 284, 288].
- [x] Model training on MNIST digits 0-4[cite: 291].
- [x] OOD evaluation using digits 5-9 and AUROC calculation[cite: 299, 301].
- [x] Image generation via sequential sampling[cite: 304].

---

## 📝 Authors
* **Instructor:** Dr. S. Amini [cite: 6, 193]
* **Phase 1 Team:** Ashkan Yousefnia, Arshak Razvani, Zahra Sorkhaei, Rojin Salmani, Mohammad Eshtehardian, Mohammad Hossein Momeni [cite: 13]
* **Phase 2 Team:** Arshak Razvani, Mohammad Eshtehardian [cite: 200]

**Department of Electrical Engineering, Sharif University of Technology** [cite: 7, 8, 194, 195]  
**Term:** Fall 1404 [cite: 9, 196]
