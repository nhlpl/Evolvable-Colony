Here is the mathematical foundation behind the Evolvable Colony v2.0 framework, covering the core algorithms and their code implementations.

---

## 📐 Mathematical Foundations of Evolvable Colony v2.0

### 1. Multi-Objective Pareto Optimization

**Goal**: Maintain a set of solutions where no objective can be improved without degrading another.

**Pareto Dominance**: Solution A dominates solution B if:
$$\forall i \in \{1..k\}: f_i(A) \ge f_i(B) \quad \text{and} \quad \exists j: f_j(A) > f_j(B)$$

**Hypervolume Metric**: Quantifies both convergence and diversity:
$$HV(S) = \lambda\left(\bigcup_{s \in S} [f_1(s), r_1] \times \cdots \times [f_k(s), r_k]\right)$$
where $r$ is a reference point and $\lambda$ is the Lebesgue measure.

```python
def dominates(self, other: 'MultiObjectiveFitness') -> bool:
    better_in_any = False
    for key in self.objectives:
        if self.objectives[key] < other.objectives[key]:
            return False
        if self.objectives[key] > other.objectives[key]:
            better_in_any = True
    return better_in_any
```

---

### 2. MAP-Elites Quality-Diversity

**Goal**: Fill a grid of behavioral niches with the highest-performing solution for each niche.

**Behavioral Binning**:
$$b_i = \left\lfloor \frac{x_i - \min_i}{\max_i - \min_i} \cdot d_i \right\rfloor$$
where $x$ is the behavior vector, $[\min, \max]$ are the behavior ranges, and $d$ is the grid dimension.

**Selection**: Each cell maintains the elite with maximum fitness:
$$E_{b} = \arg\max_{s \in \text{cell}_b} f(s)$$

```python
def _behavior_to_bin(self, behavior: np.ndarray) -> Tuple[int, ...]:
    bins = []
    for i, (low, high) in enumerate(self.ranges):
        normalized = (behavior[i] - low) / (high - low)
        bin_idx = int(np.clip(normalized * self.dims[i], 0, self.dims[i] - 1))
        bins.append(bin_idx)
    return tuple(bins)
```

---

### 3. Thompson Sampling for A/B Testing

**Goal**: Adaptively route traffic to variants based on observed performance.

**Beta-Bernoulli Model**: Each variant's success rate is modeled as:
$$P(\text{success}) \sim \text{Beta}(\alpha, \beta)$$
where $\alpha = \text{successes} + 1$, $\beta = \text{failures} + 1$.

**Thompson Sampling**: At each decision, sample from the posterior and select the variant with the highest sample:
$$a^* = \arg\max_{a} \tilde{\theta}_a \quad \text{where} \quad \tilde{\theta}_a \sim \text{Beta}(\alpha_a, \beta_a)$$

```python
alpha = stats['successes']
beta = stats['trials'] - stats['successes']
sample = np.random.beta(max(1, alpha), max(1, beta))
```

---

### 4. Memory Consolidation with Exponential Decay

**Goal**: Strengthen important memories and decay unused ones, inspired by the Ebbinghaus forgetting curve.

**Forgetting Curve**:
$$R(t) = e^{-\frac{t}{S}}$$
where $S$ is the memory strength and $t$ is time since last access.

**Reinforcement**: Each retrieval increases strength:
$$S' = S \cdot (1 + \beta)$$
where $\beta$ is the reinforcement boost.

```python
age = time.time() - ep.last_accessed
decay = max(0.1, 1.0 - self.decay_rate * age / 86400)
ep.importance = min(2.0, ep.importance + boost)
```

---

### 5. Tournament Selection

**Goal**: Select parents for reproduction with selection pressure proportional to fitness.

**Tournament Selection Probability**: For tournament size $k$, the probability of selecting the $i$-th ranked individual is:
$$P(i) = \frac{(n-i+1)^k - (n-i)^k}{n^k}$$
where $n$ is the population size and individuals are ranked $1$ (best) to $n$ (worst).

```python
indices = random.sample(range(len(agents)), min(self.tournament_size, len(agents)))
best_idx = max(indices, key=lambda i: fitnesses[i])
```

---

### 6. Gaussian Mutation for Continuous Parameters

**Goal**: Introduce small, unbiased variations to evolvable parameters.

**Gaussian Mutation**:
$$x' = x + \mathcal{N}(0, \sigma^2)$$
clamped to $[x_{\min}, x_{\max}]$.

```python
value += random.gauss(0, std)
return max(0.0, min(1.0, value))
```

---

### 7. Softmax Action Selection (Exploration-Exploitation)

**Goal**: Balance exploration of new actions with exploitation of known good actions.

**Softmax Probability**:
$$P(a_i) = \frac{e^{Q(a_i) / \tau}}{\sum_j e^{Q(a_j) / \tau}}$$
where $Q(a)$ is the estimated value of action $a$ and $\tau$ is the temperature parameter.

In our framework, this is controlled by the `temperature` parameter in `PromptGenome`.

---

### 8. Cross-Entropy Loss for LLM Fine-Tuning (Implicit)

When LLM agents generate responses, the underlying model minimizes cross-entropy between predicted and actual token distributions:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log P(y_i | x, y_{<i})$$

This is handled by the LLM API calls, not directly in our code.

---

### 9. Novelty Search (Optional Extension)

**Goal**: Reward agents for discovering novel behaviors rather than just optimizing fitness.

**Novelty Score**: Average distance to $k$-nearest neighbors in behavior space:
$$\text{novelty}(b) = \frac{1}{k}\sum_{i=1}^k \|b - b_i\|_2$$

**Selection**: $\text{fitness}_{\text{combined}} = \alpha \cdot \text{fitness} + (1-\alpha) \cdot \text{novelty}$

---

### 10. Distributed Evolution Speedup

**Goal**: Parallelize fitness evaluations across multiple workers.

**Speedup Model** (Amdahl's Law):
$$S = \frac{1}{(1-P) + \frac{P}{N}}$$
where $P$ is the parallelizable fraction and $N$ is the number of workers.

In practice, with asynchronous evaluation:
$$\text{Throughput} \approx \frac{N \cdot \text{eval\_rate}}{1 + \text{communication\_overhead}}$$

---

## 💎 Summary Table

| Component | Mathematical Foundation | Key Formula |
|:---|:---|:---|
| Pareto Optimization | Pareto Dominance | $A \succ B \iff \forall i: f_i(A) \ge f_i(B) \land \exists j: f_j(A) > f_j(B)$ |
| MAP-Elites | Behavioral Binning | $b_i = \lfloor \frac{x_i - \min_i}{\max_i - \min_i} \cdot d_i \rfloor$ |
| Thompson Sampling | Beta Posterior Sampling | $\tilde{\theta} \sim \text{Beta}(\alpha, \beta)$ |
| Memory Decay | Ebbinghaus Forgetting | $R(t) = e^{-t/S}$ |
| Tournament Selection | Rank-Based Selection | $P(i) \propto (n-i+1)^k$ |
| Gaussian Mutation | Normal Distribution | $x' \sim \mathcal{N}(x, \sigma^2)$ |
| Softmax Exploration | Boltzmann Distribution | $P(a) \propto e^{Q(a)/\tau}$ |

These mathematical foundations power the Evolvable Colony v2.0, enabling principled multi-objective optimization, quality-diversity search, adaptive A/B testing, and biologically-inspired memory consolidation.
