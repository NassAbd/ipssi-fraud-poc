# 🍎 macOS Hardware Optimization for ML & Data Dev

This guide focuses on utilizing the Unified Memory Architecture (UMA) and Metal Performance Shaders (MPS) to accelerate your local development workflow.

## 1. Accelerating TensorFlow with Metal (MPS)

By default, TensorFlow uses the CPU on Mac. To utilize the GPU, you must use the `tensorflow-metal` plug-in.

### Installation

```bash
uv add tensorflow-macos tensorflow-metal
```

### Verification

Ensure your code recognizes the GPU device.

```python
import tensorflow as tf

devices = tf.config.list_physical_devices('GPU')
print(f"GPU detected: {devices}")  # Should show 'GPU:0'
```

## 2. Resource Monitoring with ASITOP

To monitor your Apple Silicon hardware utilization—specifically CPU, GPU, Neural Engine, Memory, and Power consumption—you can use `asitop`.

### Zero-Global Installation

> **Guardian Clause Warning**: In accordance with our "Zero-Global Isolation" rule, we **do not** install CLI tools system-wide via `pip install asitop` or `brew`. 

Instead, install `asitop` as an isolated user-level tool via `uv`:

```bash
uv tool install asitop
```

### Usage

`asitop` reads Apple's `powermetrics` under the hood, so it requires root privileges to execute:

```bash
sudo asitop
```

### Interpreting the Dashboard

When running `asitop`, you will see visual dashboards separated into three major sections. Here is what they mean for your ML workflows:

1. **Core Usage (E-CPU, P-CPU, GPU, ANE)**
   - **E-CPU & P-CPU Usage:** Displays the activity level and clock speeds (MHz) divided into Efficiency and Performance CPU cores. Model preprocessing heavily hits the P-cores.
   - **GPU Usage:** Tracks utilization and frequency. If you've configured MPS correctly, you should see GPU utilization spike when training models.
   - **ANE Usage:** The Apple Neural Engine is occasionally used by specific CoreML tasks but is generally inactive during raw TensorFlow/PyTorch MPS operations.

2. **Memory (RAM)**
   - Displays used space out of system total (e.g., `14.6/48.0GB`).
   - Because of macOS's **Unified Memory Architecture (UMA)**, this memory pool is shared dynamically between the CPU and GPU.
   - Check the **swap** status (`swap inactive`). If it becomes active, you're requesting larger tensors/batches than your physical memory can handle, introducing severe I/O bottlenecks.

3. **Power & Throttling**
   - Shows realtime wattage for CPU, GPU, and total package (avg and peak).
   - Watch the `throttle: no` flag. If it changes to `yes`, the system is thermally limited, which lowers clock speeds and increases model training time.