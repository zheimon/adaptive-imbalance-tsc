# AdaCAL: Adaptive Convergence-Aware Loss for Class-Imbalanced Time-Series Classification

## Section 1: Motivation

Class imbalance in time-series classification presents a fundamental optimization challenge: gradient contributions during training are dominated by majority-class samples. If a dataset has $K$ classes with class frequencies $n_1 \geq n_2 \geq \cdots \geq n_K$, the empirical risk minimized during training is:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)$$

where $N = \sum_k n_k$. Each gradient step is thus dominated by the majority class, since $\mathbb{E}[\nabla_\theta \ell \mid y = c] \propto n_c / N$.

Standard reweighting methods address this by assigning static weights derived from class frequencies, e.g., inverse frequency weighting $w_c = N / (K \cdot n_c)$ or the effective number weighting of Cui et al. (2019). While effective as a first-order correction, these methods have a critical limitation: **they are static**. They ignore whether minority classes are actually learning, whether the model has stagnated on certain classes, or whether the validation performance gap has widened during training.

We propose **AdaCAL** (Adaptive Convergence-Aware Loss), a method that uses *online training signals*—per-class loss trajectories, gradient norm dynamics, and validation F1 gaps—to dynamically adjust per-class loss weights throughout training. AdaCAL amplifies the loss signal for classes exhibiting convergence problems, enabling the model to escape local minima caused by gradient dominance.

---

## Section 2: Formal Definition of "Convergence-Aware"

Let $L_c(t)$ denote the mean training loss on class $c$ samples at epoch $t$, computed over all training samples with label $c$.

### 2.1 Loss Plateau

Class $c$ is in **plateau** at epoch $t$ if the relative change in class loss over a lookback window of $k$ epochs is below threshold $\tau_\text{plateau}$:

$$\delta_c(t) = \frac{|L_c(t) - L_c(t-k)|}{L_c(t-k) + \varepsilon} < \tau_\text{plateau}$$

where $\varepsilon = 10^{-8}$ prevents division by zero and $\tau_\text{plateau} = 0.01$ (1% relative change). Default lookback $k = 5$ epochs. A class in plateau has stopped improving despite ongoing training, indicating either gradient starvation or a local minimum.

### 2.2 Gradient Stagnation

Let $\|\nabla_\theta L_c(t)\|_2$ be the $\ell_2$-norm of the gradient of class $c$'s loss with respect to model parameters $\theta$. Define the normalized gradient as:

$$\tilde{G}_c(t) = \frac{\|\nabla_\theta L_c(t)\|_2}{\bar{G}(t)}, \quad \bar{G}(t) = \frac{1}{K}\sum_{c=1}^{K} \|\nabla_\theta L_c(t)\|_2$$

Class $c$ exhibits **gradient stagnation** if:

$$G_c(t) = \tilde{G}_c(t) < \tau_\text{grad}$$

with default $\tau_\text{grad} = 0.5$. A normalized gradient below 0.5 means the class is contributing less than half the average gradient magnitude, indicating its signal is being drowned out by other classes.

### 2.3 Validation F1 Gap

Let $F1_c(t)$ denote the validation F1 score for class $c$ at epoch $t$, and let $m = \arg\max_c F1_c(t)$ be the best-performing class. The **F1 gap** for class $c$ is:

$$F_c(t) = F1_m(t) - F1_c(t) \geq 0$$

A positive gap means class $c$ is underperforming relative to the best class. This provides a direct measure of the generalization imbalance, complementing the training-domain signals above.

---

## Section 3: AdaCAL Weight Update Rule

AdaCAL maintains per-class adaptive weights $\mathbf{w}(t) = [w_1(t), \ldots, w_K(t)]$, initialized to $w_c(0) = 1$ for all $c$.

### 3.1 Composite Signal

The composite score for class $c$ at epoch $t$ is:

$$s_c(t) = \alpha_1 \cdot \text{plateau\_score}_c(t) + \alpha_2 \cdot \text{grad\_score}_c(t) + \alpha_3 \cdot \text{f1\_gap\_score}_c(t)$$

with hyperparameters $\alpha_1 = 0.4$, $\alpha_2 = 0.3$, $\alpha_3 = 0.3$.

Individual scores are defined as:

- **Plateau score**: $\text{plateau\_score}_c(t) = \mathbf{1}[\delta_c(t) < \tau_\text{plateau}]$, indicating binary plateau state.
- **Gradient score**: $\text{grad\_score}_c(t) = \max(0,\ \tau_\text{grad} - \tilde{G}_c(t))$, positive only for stagnant classes.
- **F1 gap score**: $\text{f1\_gap\_score}_c(t) = \frac{F_c(t) - \overline{F}(t)}{\sigma_F(t) + \varepsilon}$, the standardized F1 gap.

### 3.2 Zero-Sum Normalization

Before applying the update, scores are **zero-sum normalized**:

$$\tilde{s}_c(t) = s_c(t) - \frac{1}{K}\sum_{c'=1}^{K} s_{c'}(t)$$

This ensures $\sum_c \tilde{s}_c(t) = 0$, so the total weight mass is preserved under the exponential update.

### 3.3 Exponential Weight Update

The per-class weight is updated multiplicatively:

$$w_c(t+1) = w_c(t) \cdot \exp\!\bigl(\eta \cdot \tilde{s}_c(t)\bigr)$$

with learning rate $\eta = 0.1$. After updating, weights are renormalized to maintain a stable scale:

$$w_c \leftarrow \frac{w_c}{\frac{1}{K}\sum_{c'} w_{c'}}$$

so that $\frac{1}{K}\sum_c w_c = 1$ at all times (mean weight is 1, sum is $K$).

### 3.4 Loss Computation

The AdaCAL loss at epoch $t$ is the weighted cross-entropy:

$$\mathcal{L}_\text{AdaCAL}(\theta; t) = \frac{1}{N}\sum_{i=1}^{N} w_{y_i}(t) \cdot \ell_\text{CE}(f_\theta(x_i), y_i)$$

---

## Section 4: Connection to Effective Sample Size

Cui et al. (2019) define the effective number of samples for class $c$ with $n_c$ instances as:

$$E_c = \frac{1 - \beta^{n_c}}{1 - \beta}$$

and propose weighting classes by $1/E_c$. AdaCAL generalizes this by treating the effective sample size as a dynamic quantity. When class $c$'s weight $w_c(t)$ increases, the expected gradient contribution from class $c$ scales as:

$$\mathbb{E}\!\left[\|\nabla_\theta \mathcal{L}_\text{AdaCAL}\|_2 \mid y = c\right] \propto w_c(t) \cdot \frac{n_c}{N}$$

Thus $w_c(t)$ implicitly defines a **dynamic effective count** $\tilde{n}_c(t) = w_c(t) \cdot n_c$, which AdaCAL adjusts based on observed convergence behavior. Classes that exhibit plateau, gradient stagnation, or a growing F1 gap receive increased effective counts, steering the optimizer toward under-learned regions of the loss landscape. This connects AdaCAL to the theoretical motivation of effective number reweighting while transcending its static limitation.

---

## Section 5: Computational Complexity

AdaCAL introduces negligible overhead relative to the base model forward/backward pass.

**Storage**: $O(K \cdot k)$ for the loss trajectory buffer (one scalar per class per lookback epoch), $O(K)$ for gradient norm and F1 accumulators. For $K = 100$ classes and $k = 5$, this is 500 floats — trivially small.

**Computation per epoch**:
- Per-class loss accumulation: $O(N)$ (one pass through training data, no extra operations).
- Gradient norm computation: requires one backward pass on a small probe batch of size $B_\text{probe}$ per class, costing $O(K \cdot B_\text{probe} \cdot P)$ where $P$ is the number of model parameters. With $B_\text{probe} = 32$, this adds at most 10–20% overhead for typical TSC models.
- Weight update: $O(K)$ arithmetic operations.

**Total extra cost**: $O(K \cdot B_\text{probe} \cdot P + N)$ per epoch — dominated by the probe backward passes, but still linear in $K$ and negligible compared to the full training epoch cost of $O(N \cdot P)$ for large $N$.

AdaCAL thus provides convergence-aware adaptive reweighting with essentially no additional hyperparameter sensitivity beyond $\eta$, $\alpha_1, \alpha_2, \alpha_3$, and $k$, all of which have principled defaults derived from the signal definitions above.
