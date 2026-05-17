# TC Metric Reference

All metrics follow the convention: **higher = more jitter**.  
`1-SER` is listed in place of raw SER so all metrics point the same direction.

---

## Metric Definitions

### MACD — Mean Absolute Consecutive Difference

$$\text{MACD} = \frac{1}{(T-1) \cdot J} \sum_{t=1}^{T-1} \sum_{j=1}^{J} |a_j[t] - a_j[t-1]|$$

Average step-to-step change in action values, averaged across all joints and timesteps.  
Simple and interpretable — directly measures how much the commanded position jumps between consecutive timesteps.

---

### Jerk — Normalised Jerk

$$\text{Jerk} = \frac{1}{J} \sum_{j=1}^{J} \frac{1}{T-3} \sum_t \left| \frac{\Delta^3 a_j[t]}{\Delta t^3} \right| \cdot \frac{1}{\text{range}(a_j)}$$

Third finite difference of the action signal (rate of change of acceleration), normalised by the signal range to be scale-independent.  
Physically meaningful — jerk is what causes mechanical stress in robot joints and perceptible "lurching" motion.

---

### DCC — Direction Change Count

$$\text{DCC} = \frac{1}{J} \sum_{j=1}^{J} \frac{\text{# sign changes in } \Delta a_j}{T - 2}$$

Fraction of timesteps where the direction of motion reverses for each joint, averaged across joints.  
Captures oscillatory back-and-forth behaviour that other frequency-domain metrics may miss when the amplitude is small.

---

### SE — Spectral Entropy

$$\text{SE} = -\frac{1}{\log_2(N/2+1)} \sum_f p(f) \log_2 p(f), \quad p(f) = \frac{|F(f)|^2}{\sum|F(f)|^2}$$

Shannon entropy of the normalised power spectrum, scaled to [0, 1].  
High SE means energy is spread across many frequencies (noise-like signal). Low SE means energy is concentrated at a few frequencies (smooth, periodic motion).

---

### Sm — Weighted Mean Frequency

$$\text{Sm} = \frac{2}{N \cdot f_s} \sum_f |F(f)| \cdot f$$

Amplitude-weighted mean of the frequency content.  
The primary metric developed for this project. Higher Sm means the action signal has more energy at high frequencies, indicating jitter. Used as the main TC score throughout training evaluation.

---

### SER — Signal Energy Ratio  (reported as 1 − SER)

$$\text{SER}_\text{raw} = \frac{\sum_{f \leq f_\text{thresh}} |F(f)|^2}{\sum_f |F(f)|^2}, \quad f_\text{thresh} = 1.0 \text{ Hz}$$

Fraction of signal energy below a threshold frequency (1 Hz here).  
Raw SER is higher for smoother signals, so `1-SER` is reported to keep the convention that higher = more jitter.  
Very sensitive to the threshold choice; 1 Hz was selected to match the typical frequency of intentional robot motion.

---

### ADR — Autocorrelation Decay Rate

Fits an exponential decay to the normalised autocorrelation function over lags 0 to min(T/2, 50):

$$R(\tau) \approx e^{-\lambda \tau} \implies \lambda = -\frac{d}{d\tau} \log R(\tau)$$

Returns the decay constant λ (s⁻¹). A smooth signal stays correlated over long lags (low λ); a jittery signal loses correlation quickly (high λ).  
In practice this metric performed poorly on the real episodes (see ranking table below).

---

## Recorded Metric Values

### Table 1 — Full 7-metric comparison (real + synthetic datasets)

Source: `compare_tc_metrics.py` run on `BrandonAL/smooth_and_jittery_so101`  
Synthetic signal: sine wave (0.5–1.5 Hz per joint) + white noise at four σ levels  
Convention: higher = more jitter for all metrics

| Dataset                  |   MACD  |    Jerk  |   DCC   |    SE   |    Sm   |  1-SER  |   ADR   |
|--------------------------|--------:|---------:|--------:|--------:|--------:|--------:|--------:|
| real\_smooth (ep0)       | 0.28851 |  16.632  | 0.00818 | 0.31982 | 0.00347 | 0.00069 | 0.32784 |
| real\_jittery (ep1)      | 0.41298 |  53.367  | 0.06564 | 0.30811 | 0.00477 | 0.00891 | 0.30533 |
| synthetic\_clean (σ=0)   | 0.13312 | 105.627  | 0.06693 | 0.14546 | 0.00067 | 0.49340 | 0.29553 |
| synthetic\_low (σ=0.1)   | 0.16850 | 3806.442 | 0.35843 | 0.17613 | 0.00228 | 0.50146 | 0.29914 |
| synthetic\_med (σ=0.5)   | 0.58620 |10837.195 | 0.64558 | 0.50775 | 0.01013 | 0.64401 | 0.29434 |
| synthetic\_high (σ=2.0)  | 2.32036 |15862.666 | 0.68173 | 0.89345 | 0.04061 | 0.88780 | 0.61639 |
| **Discrim. power D**     |  +0.431 |  +2.209  | +7.026  | -0.037  | +0.373  | +11.994 | -0.069  |
| **Monotone (synthetic)** |   YES   |   YES    |   YES   |   YES   |   YES   |   YES   |   NO    |

D = (jitter_score − smooth_score) / (|smooth_score| + ε).  Higher D = better separation.

---

### Table 2 — Metric ranking by discriminative power (real episodes)

| Rank | Metric | D score | Monotone on synthetic |
|-----:|--------|--------:|:---------------------:|
|  1   | 1-SER  | +11.994 | YES |
|  2   | DCC    |  +7.026 | YES |
|  3   | Jerk   |  +2.209 | YES |
|  4   | MACD   |  +0.431 | YES |
|  5   | Sm     |  +0.373 | YES |
|  6   | SE     |  -0.037 | YES |
|  7   | ADR    |  -0.069 | NO  |

**Sm** (the primary project metric) ranks 5th. It is monotone on synthetic data and correctly identifies jittery > smooth on real data, but with lower discriminative power than 1-SER, DCC, and Jerk.

---

### Table 3 — Baseline vs LowPass vs AJAS (1 episode, libero\_object task 0)

Source: `baseline_1ep/`, `lowpass_1ep/`, `ajas_1ep/` eval runs  
Task: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`  
Note: only Sm and SER were recorded in these runs (full 7-metric logging not yet implemented)  
Note: T=10 frames — very short episode, treat these values with caution

| Condition                       | Sm (raw) | Sm (post-proc) | SER (raw)  | SER (post-proc) | Success |
|---------------------------------|---------:|---------------:|-----------:|----------------:|:-------:|
| Baseline (no processor)         | 0.009076 |      —         | ~0         |       —         |  0/1    |
| LowPass (3 Hz, order 2)         | 0.008835 |     0.003765   | ~0         |      ~0         |  0/1    |
| AJAS (SER threshold 1 Hz)       | 0.004741 |     0.004741   | ~0         |      ~0         |  0/1    |

LowPass reduced Sm by **58%** relative to baseline on the processed output.  
AJAS raw Sm is already lower than baseline, suggesting the action selection itself changed.  
SER values are effectively zero across all conditions — the very short episode (T=10) means there is insufficient frequency resolution for a meaningful SER estimate.

---

*Last updated: 2026-04-22*
