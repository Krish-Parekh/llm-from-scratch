# Loss Functions

## Loss vs Cost

**Loss function** — computed for a single training example.  
**Cost function** — average (or sum) of the loss over the whole batch.

---

## Quick decision guide

```
Is your target a category or a number?
├── Number (continuous)          → Regression
│   ├── Clean data               → MSE
│   ├── Many outliers            → MAE
│   └── Some outliers            → Huber
└── Category (class label)       → Classification
    ├── 2 classes (0/1)          → Binary Cross Entropy + sigmoid
    ├── 3+ classes               → Categorical Cross Entropy + softmax
    │   ├── labels: 0, 1, 2...   → sparse_categorical_crossentropy
    │   └── labels: one-hot      → categorical_crossentropy
    └── Multiple labels per sample → BCE per label (multi-label)
```

---

## MSE / L2 Loss

**Formula:** `(Yi - Yhat)²`

Used in **regression** problems.

- Why not just `(true - pred)`? If we have `[0.2, -0.2, 4, -4]`, positive and negative errors cancel out and don't reflect true error.
- Squaring penalizes larger errors more aggressively, which can cause drastic weight updates.

### Advantages

- Easy to work with mathematically
- Differentiable — works well with gradient descent
- Single local minimum (smooth bowl-shaped loss surface)

### Disadvantages

- Error is in squared units (e.g. predicting salary off by 5 lakhs → loss of 25)
- Not robust to outliers — large errors dominate training

### When to use (dataset examples)

- House price prediction on relatively clean data (no extreme mansions skewing everything)
- Predicting exam scores, temperature, or stock returns when outliers are rare
- When you assume errors are roughly normally distributed (Gaussian noise)

### When NOT to use (dataset examples)

- Salary prediction where a few CEOs earn 100× more than everyone else
- Sensor data with occasional bad readings (faulty device spikes)
- When you need the error reported in the same unit as the target (use MAE instead)

---

## MAE / L1 Loss

**Formula:** `|Yi - Yhat|`

Used in **regression** problems.

- Without absolute value, positive and negative errors cancel out (same problem as above).
- Unlike MSE, errors are penalized linearly — off by 10 counts as 10, not 100.

### Advantages

- Error is in the same unit as the target (e.g. salary in lakhs → MAE in lakhs)
- More robust to outliers than MSE
- Differentiable everywhere except at zero (optimizers handle this in practice)

### Disadvantages

- Not differentiable exactly at zero (`|x|` has a sharp corner)
- Gradient is constant (+1 or −1) regardless of error size — no stronger signal for very wrong predictions
- Can have multiple local minima

### When to use (dataset examples)

- Predicting delivery time when a few orders are extremely late
- House rent in a city with a few ultra-luxury listings skewing prices
- Any regression where you want median-like behavior and errors in original units

### When NOT to use (dataset examples)

- Clean, well-behaved datasets where MSE converges faster (e.g. physics simulations with low noise)
- When you want the model to heavily punish large mistakes (e.g. medical dosage — sometimes MSE or Huber is better)
- When you need the smoothest possible loss landscape for optimization

---

## Huber Loss

**Formula:**

- If `|Yi - Yhat| ≤ delta` → `0.5 × (Yi - Yhat)²`
- If `|Yi - Yhat| > delta` → `delta × |Yi - Yhat| - 0.5 × delta²`

Used in **regression** when you want a middle ground between MSE and MAE.

- Small errors (within `delta`) → behaves like MSE (smooth, good gradients)
- Large errors (beyond `delta`) → behaves like MAE (outliers don't dominate)
- `delta` is a hyperparameter (common default: 1.0)

### Advantages

- Best of both worlds: smooth like MSE for small errors, robust like MAE for large errors
- Widely used when data has some noise or outliers (robotics, finance)

### Disadvantages

- Need to tune `delta` — too small acts like MAE, too large acts like MSE
- More complex than plain MSE or MAE
- Harder to interpret directly (mixes squared and linear terms)

### When to use (dataset examples)

- Real estate prices — mostly normal houses but a few expensive outliers
- Robotics / self-driving — sensor readings mostly accurate but occasionally spike
- Financial data (stock returns) — mostly small moves, occasional crash days
- Any regression dataset with *some* outliers but not completely messy

### When NOT to use (dataset examples)

- Perfectly clean data with no outliers — just use MSE
- Heavily corrupted data where most points are outliers — MAE alone may be safer
- When you don't want to tune an extra hyperparameter

---

## Binary Cross Entropy (Log Loss)

**Formula:** `-[Yi × log(Yhat) + (1 - Yi) × log(1 - Yhat)]`

- `Yi` = true label (0 or 1)
- `Yhat` = predicted probability (0 to 1, from **sigmoid** activation)

Used in **binary classification** — predicting one of two classes.

- The log term heavily punishes confident wrong predictions (e.g. predicts 0.99 for class 0 when truth is 1).
- MSE treats 0 and 1 as numbers and doesn't properly penalize uncertain probability outputs.
- Output layer must use **sigmoid**. Labels must be 0 or 1.

### Advantages

- Designed for probability outputs — rewards confident correct predictions
- Smooth and differentiable (when predictions are clipped away from 0 and 1)
- Mathematically correct loss for Bernoulli distribution

### Disadvantages

- Only works for exactly 2 classes
- Sensitive to class imbalance without class weights
- Loss values not intuitive in real-world units

### When to use (dataset examples)

- Email spam detection (spam = 1, not spam = 0)
- Fraud detection (fraudulent = 1, legitimate = 0)
- Disease diagnosis (has disease = 1, healthy = 0)
- Click prediction (clicked = 1, didn't click = 0)

### When NOT to use (dataset examples)

- MNIST digit classification (10 classes) — use categorical / sparse categorical cross entropy
- Iris flower species (3 classes) — use categorical / sparse categorical cross entropy
- Predicting house price (continuous) — use MSE/MAE/Huber
- Multi-label problems (cat AND dog both present) — use BCE per label separately
- Heavily imbalanced datasets without class weights

---

## Categorical Cross Entropy

**Formula:** `-Σ (Yi × log(Yhat_i))` summed over all classes

- `Yi` = true label as **one-hot vector** (e.g. `[0, 0, 1, 0]` for class 2)
- `Yhat_i` = predicted probability for class `i` (from **softmax** — all classes sum to 1)

Used in **multi-class classification** — one class out of 3 or more (mutually exclusive).

- Extension of binary cross entropy to multiple classes — only the true class contributes to the loss.
- Output layer must use **softmax**.
- Keras: `loss='categorical_crossentropy'`
- Labels must be **one-hot encoded** before training.

### Advantages

- Natural fit for multi-class problems where each sample belongs to exactly one class
- Mathematically correct loss for categorical (multinomial) distribution
- Heavily punishes confident wrong class predictions

### Disadvantages

- Requires one-hot encoding labels — extra preprocessing step
- Memory overhead with many classes (e.g. 10,000-class vocab → huge one-hot vectors)
- Only for mutually exclusive classes (exactly one label per sample)
- Suffers from class imbalance without class weights

### When to use (dataset examples)

- Iris flower classification when you've manually one-hot encoded the 3 species
- Small multi-class problems where you already have one-hot labels from a pipeline
- When labels are naturally stored as probability distributions (soft labels)

### When NOT to use (dataset examples)

- MNIST with integer labels `0–9` — use `sparse_categorical_crossentropy` instead (no need to one-hot)
- Binary problems with only 2 classes — binary cross entropy is simpler
- Regression (house price, temperature) — use MSE/MAE/Huber
- Large vocabulary problems (LLM token prediction with 50k+ classes) — one-hot is impractical, use sparse version

---

## Sparse Categorical Cross Entropy

**Formula:** `-log(Yhat_k)` where `k` is the true class index

- Equivalent math to categorical cross entropy — just skips the one-hot step internally
- `k` = true label as an **integer** (e.g. `2` means class 2)
- `Yhat_k` = predicted probability for class `k` (from **softmax**)

Used in **multi-class classification** when labels are stored as integer class indices.

- Keras: `loss='sparse_categorical_crossentropy'`
- Output layer still uses **softmax** (same as categorical cross entropy)
- Framework internally picks the log probability of the correct class index — no need to convert labels to one-hot

```python
# MNIST example — labels are integers 0–9, no one-hot needed
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Why sparse vs categorical?

| | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
|---|---|---|
| Label format | One-hot `[0,0,1,0,...]` | Integer `2` |
| Preprocessing | Must one-hot encode | Use labels as-is |
| Memory | High (N × num_classes) | Low (N × 1) |
| Math | Identical | Identical |

### Advantages

- No one-hot encoding step — simpler data pipeline
- Much more memory efficient with large number of classes (e.g. 10,000 token vocabulary)
- Same training behavior as categorical cross entropy
- Most built-in datasets return integer labels (MNIST, CIFAR, etc.)

### Disadvantages

- Only works when labels are integer indices — can't handle soft labels or probability distributions directly
- Same limitations as categorical CE: mutually exclusive classes only, class imbalance issues
- Treats all misclassifications equally

### When to use (dataset examples)

- **MNIST** — labels are integers `0–9` (used in `llm/ann.ipynb`)
- **CIFAR-10** — 10 object classes as integers
- **Iris dataset** — 3 species encoded as 0, 1, 2
- **Language model next-token prediction** — token IDs as integer indices over large vocab
- Any dataset where `y_train` looks like `[3, 0, 7, 1, ...]` instead of one-hot matrices

### When NOT to use (dataset examples)

- Labels are already one-hot or soft probability distributions — use `categorical_crossentropy`
- Binary classification (2 classes) — binary cross entropy is more direct
- Regression — use MSE/MAE/Huber
- Multi-label (multiple true classes per sample) — use BCE per label
- Heavily imbalanced datasets without class weights

---

## Loss function comparison

| Loss | Type | Activation | Label format | Use when |
|------|------|------------|--------------|----------|
| MSE | Regression | linear | continuous number | Clean data, few outliers |
| MAE | Regression | linear | continuous number | Outliers, interpretable units |
| Huber | Regression | linear | continuous number | Mix of clean data + some outliers |
| Binary CE | Classification (2 classes) | sigmoid | 0 or 1 | Spam, fraud, yes/no |
| Categorical CE | Classification (3+ classes) | softmax | one-hot vector | Small multi-class, soft labels |
| Sparse Categorical CE | Classification (3+ classes) | softmax | integer index | MNIST, CIFAR, large vocab |
