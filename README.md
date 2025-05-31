# Triton Softmax — A Custom GPU Kernel in Jupyter

This project implements and benchmarks a **custom softmax GPU kernel using [Triton](https://github.com/openai/triton)** — all developed and explained inside a single Jupyter Notebook.

It explores GPU parallelism concepts like **blocks, threads, warps, vectorization, and tiling**, and compares Triton performance to native PyTorch.

---

## 🧪 What You'll Learn

- ✅ How to write a GPU kernel from scratch in Triton
- ✅ Understand thread-block architecture of GPUs
- ✅ Implement softmax with full control over memory access
- ✅ Use vectorized loads and tiling for speed
- ✅ Benchmark Triton vs PyTorch

---

## 🔍 Key Techniques Used

| Technique        | Used In Kernel            |
|------------------|---------------------------|
| Naive parallelism| Basic softmax kernel      |
| Vectorized loads | `VEC=4` optimization      |
| Tiling           | Handles large input sizes |
| Multi-pass logic | Ensures accuracy          |

---


## 📊 Benchmark Results (1024 × 4096 input)

| Kernel Variant        | Triton Time (ms) | PyTorch Time (ms) | Speedup vs PyTorch | Max Diff vs PyTorch   |
| --------------------- | ---------------- | ----------------- | ------------------ | --------------------- |
| 🧪 Naive              | 0.2310           | 0.4021            | **1.74× faster**   | 3.73e-09 ✅            |
| ⚡ Vectorized          | 0.0655           | 0.4000            | **6.11× faster**   | 5.01e+00 ❌ inaccurate |
| 🧠 Vectorized + Tiled | 0.3728           | 0.4187            | **1.12× faster**   | 2.15e-02 ⚠️ better    |


## 📁 Files

- `GPU-Kernel using Triton.ipynb.ipynb` — the notebook containing:
  - All three kernel implementations
  - Full benchmark and explanation
  - Visual breakdown of GPU behavior

---

## ▶️ How to Run

1. **Install Triton** (Python ≥3.8, CUDA-enabled GPU):
   ```bash
   pip install triton
   ```

2. **Launch the notebook**:
   ```bash
   jupyter notebook GPU_Kernel_using_Triton.ipynb
   ```

3. The notebook:
   - Runs all kernels
   - Benchmarks against PyTorch
   - Displays runtime and accuracy metrics

---

## 🧠 Why This Matters

This notebook shows how GPU performance comes not just from "writing a kernel," but from:
- Choosing the right block size
- Avoiding warp divergence
- Using multi-pass reductions
- Leveraging vectorization and memory reuse

---

## 🙋‍♀️ Author

Built by Maria Pushparaj as a 1-month challenge to master GPU programming with Triton.  
If you're hiring for ML infra, compilers, or performance roles — let’s connect!

---

