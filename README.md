# Tri-Network-Synaptic-Routing
Official implementation of Tri-Network Synaptic Routing architecture for adversarially immune, replay-free continual learning.
# Tri-Network Synaptic Routing: Mitigating Catastrophic Forgetting

Official open-source reference implementation and validation engine for the **Tri-Network Synaptic Routing** model architecture[cite: 1]. This framework decouples representational plasticity from stability matrices through hard synaptic gating constraints using a completely non-parametric metric engine[cite: 1].

## 📄 Core Scientific Manuscript
* **Permanent DOI Pre-print Archive:** https://doi.org/10.5281/zenodo.20708945
* **Citation Registration ID:** `10.5281/zenodo.20708945`
* **License Matrix:** Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

---

## 🚀 The Low-Compute Paradigm
Unlike heavy deep learning approaches that depend on multi-million dollar institutional GPU server blocks, this architecture is mathematically designed for resource-constrained accessibility[cite: 1]:
* **Development Hardware:** Apple MacBook Air (Local MPS Pipeline Execution)[cite: 1]
* **Cloud Infrastructure Allocation:** Google Colab (Free Open Public Tier)[cite: 1]

---

## 📊 Core Empirical Benchmarks

This pipeline implements a series of stress tests to evaluate model degradation across highly conflicting sequential inputs[cite: 1]:

| Benchmark Objective | Performance Metrics (Mean ± Std) | Operational Overhead Status |
| :--- | :--- | :--- |
| **Split-MNIST Task 0 Retention** | **99.67%**[cite: 1] | 0 Replay Buffers Employed[cite: 1] |
| **CIFAR-100 10-Task Step Mean** | **94.88% ± 0.41%**[cite: 1] | 0 Gating Layer Weights Trained[cite: 1] |
| **Adversarial Poisoning Attack** | **0.00 ± 0.48 pp Degradation**[cite: 1] | Complete Parameter Neighborhood Isolation[cite: 1] |
| **Visually Similar Sibling Domains** | **74.60% Unsupervised Accuracy**[cite: 1] | High-Difficulty Boundary Delineation[cite: 1] |
| **Per-Task Parameter Growth Footprint**| **0.2803% Expansion Overhead**[cite: 1] | Sub-linear System Expansion (2.80% Total)[cite: 1] |

---

## 🛠️ Quick Start (Google Colab / Local Setup)

To execute our cross-validation suite inside a browser instance without setting up a local environment, click the master notebook file you upload here, open it directly inside a **Google Colab standard instance**, change your runtime execution chip configuration to **T4 GPU**, and run all cells.

For local deployment, verify that your active workspace context satisfies the core dependency requirements:

```bash
pip install torch torchvision numpy
