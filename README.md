# ARGUS SHIELD

> **"Don't just trust what your model sees. Verify that it deserves to be trusted."**

ARGUS SHIELD is a computer-vision robustness and anomalous-prediction-detection layer. It evaluates how AI classifiers (specifically PyTorch `InceptionV3`) behave under controlled input transformations, computes risk scores based on multi-view prediction stability, and prevents over-trusting unstable predictions through consensus recovery or abstention.

---

## 1. Problem Statement & Solution

Modern deep convolutional neural networks (such as InceptionV3) can achieve high accuracy on standard benchmarks, but remain vulnerable to mild input perturbations or adversarial manipulations. A model may output high confidence (e.g. 91% confidence on **Ostrich** for a perturbed **Tiger Cat** image), misleading downstream systems.

**ARGUS SHIELD** solves this by inserting a defense layer between raw model predictions and downstream decision systems:
- **Shield OFF**: Raw predictions are accepted blindly without stability verification.
- **Shield ON**: ARGUS analyzes prediction consistency across 7 semantically-preserving input transformations, calculates an anomalous risk score (0–100), and mitigates instability via consensus recovery or abstention.

---

## 2. Architecture & Pipeline

```
IMAGE
  ├── 1. Validation (dimensions, channels, corruption checks)
  ├── 2. Base Inference (InceptionV3 raw top-1 & top-5 predictions)
  ├── 3. Multi-View Generation (Blur, JPEG, Resize, Brightness, Contrast, Noise, Sharpen)
  ├── 4. Multi-View Inference (InceptionV3 predictions across transformed views)
  ├── 5. Consistency Analysis (agreement ratio, class switching, mean/std confidence)
  ├── 6. Risk Engine (0-100 anomalous risk score & triggered explanations)
  └── 7. Mitigation Policy (LOW -> Accept | MEDIUM -> Recheck | HIGH -> Consensus / ABSTAIN)
```

---

## 3. Adversarial Generation & Methods

The **Adversarial Lab** enables bounded, reproducible adversarial evaluation against local InceptionV3:
- **Iterative Target Class Generator (Default)**: Minimizes cross-entropy loss toward target class (e.g., Ostrich, ImageNet class 9) over $N$ iterations within $\epsilon$-box.
- **Targeted One-Step Generator**: Single targeted gradient step.
- **Basic Iterative Method (BIM)**: Untargeted iterative gradient step.
- **Fast Gradient Sign Method (FGSM)**: Single untargeted gradient step.

All parameters (epsilon budget, iteration count, target class) are strictly validated and bounded server-side in `config.py`.

---

## 4. Ground-Truth Rule & Defense Scenarios

ARGUS does **not** rely on ground-truth label shortcuts or hardcoded fixture logic:
- Raw predictions are computed via live PyTorch InceptionV3 inference.
- Multi-view views are computed via real image transformations and real model re-evaluations.
- Risk and mitigation decisions are derived strictly from computed consistency metrics.

---

## 5. Installation & Running Locally

### Prerequisites
- Python 3.10+ (PyTorch + torchvision installed)
- Node.js v18+ & npm

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run pytest backend test suite
python -m pytest tests -v

# Start FastAPI server (runs on http://127.0.0.1:8000)
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite dev server (runs on http://localhost:5173)
npm run dev
```

---

## 6. End-to-End Demo Click-Through

1. Open http://localhost:5173.
2. **Tab 1: Adversarial Lab**:
   - Select **Benchmark Tiger Cat** fixture.
   - Click **Generate Adversarial Image** (Target: Ostrich, ε = 0.03, 10 iterations).
   - Observe **Tiger Cat -> Ostrich** transition and 10x amplified difference visualization.
   - Click **Send to ARGUS Test &rarr;**.
3. **Tab 2: ARGUS Test**:
   - Toggle **Shield OFF**: Observe raw **Ostrich** prediction accepted unconditionally.
   - Toggle **Shield ON**: Observe 7-stage pipeline run across 7 transformation views, computing **HIGH Threat Level** (Risk Score > 65), triggering **CONSENSUS_RECOVERED** (Tiger Cat) or **ABSTAIN**.

Alternatively, click **Run Automated Demo Script** in the header to execute the complete pipeline automatically.

---

## 7. Technical Limitations & Research Reference

- **Model Scope**: Prototype configured for InceptionV3 (ImageNet weights).
- **Scope Statement**: This is a defensive computer vision robustness research tool. It evaluates classifier stability under input perturbations and refuses over-trusting unstable predictions. It does not attack external systems or claim to infer attacker intent.
- **Reference**: Modernized implementation based on concepts from *"Generating Adversarial Examples using PyTorch"* (PyTorch tutorials).
