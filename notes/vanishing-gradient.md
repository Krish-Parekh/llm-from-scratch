# Vanishing Gradient

During backpropagation in deep networks, gradients are multiplied layer by layer. If each is small, the product shrinks toward zero — weight updates become negligible and learning stalls.

**Example:** `0.1 × 0.1 × 0.1 = 0.001`

---

## When it happens

- **Deep networks** — more layers = more multiplications = smaller gradients
- **Sigmoid / Tanh activations** — derivatives are ≤ 1 (sigmoid max 0.25), so gradients shrink at every layer

---

## How to identify

- **Loss** plateaus early — barely changes across epochs
- **Weights** stop updating — values stay nearly constant

---

## How to fix

| Fix | Notes |
|-----|-------|
| **ReLU** | Derivative is 1 for positive inputs — gradients don't shrink. Trade-off: dying ReLU (neurons stuck at 0) |
| **Weight initialization** | Xavier (sigmoid/tanh) or He (ReLU) — start weights in a range that keeps gradients flowing |
| **Batch normalization** | Normalizes layer inputs, stabilizes gradient scale across layers |
| **Residual connections** | Skip connections let gradients bypass layers directly (ResNet) |

**Not ideal:** Reducing network depth/complexity — avoids the problem but limits pattern learning.

---

## Exploding gradient (opposite problem)

Gradients grow instead of shrink — common in **RNNs** on long sequences.

**Fixes:** gradient clipping, gated activations (LSTM, GRU), proper initialization.
