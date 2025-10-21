# Spectra AI Mini Challenge: Anomaly Prompt Detection

A proof-of-concept prototype for detecting anomalous or malicious prompts submitted to language models using statistical methods and linear algebra.

## 📋 Project Overview

This project implements an AI security system that monitors prompts submitted to language models to identify potentially malicious or anomalous inputs. Each prompt is represented as a high-dimensional embedding vector, and we use mathematical techniques to distinguish normal prompts from outliers.

## 🎯 Objectives

The prototype demonstrates:
1. **Linear Algebra**: Computation of covariance matrices and Mahalanobis distances
2. **Probability Theory**: Chi-square distribution for statistical anomaly detection
3. **Bayesian Analysis**: Posterior probability calculation for flagged prompts
4. **Visualization**: PCA-based dimensionality reduction and decision boundary plotting

## 🔧 Technical Implementation

### Key Components

#### 1. Mahalanobis Distance Calculation
- Measures how many standard deviations a prompt is from the center of the normal distribution
- Accounts for correlations between embedding dimensions
- Formula: `D_M(x) = sqrt((x - μ)ᵀ * Σ⁻¹ * (x - μ))`

#### 2. Chi-Square Based Anomaly Flagging
- Squared Mahalanobis distances follow χ²(k) distribution
- Computes p-values for each prompt
- Flags prompts with p-value < α (0.01) as anomalous

#### 3. Bayesian Posterior Probability
- Calculates P(Malicious | Flagged) using Bayes' theorem
- Accounts for prior malicious rate (5%) and detection accuracy (95%)
- Provides realistic assessment of flagged prompt risk

## 📊 Results

### Performance Metrics
- **Accuracy**: 99.4%
- **Precision**: 93.5%
- **Recall**: 100%
- **F1-Score**: 96.6%

### Key Findings
- Successfully detected all 100 anomalous prompts (100% recall)
- Only 7 false positives out of 1000 normal prompts (0.7% FPR)
- 83.3% posterior probability that a flagged prompt is truly malicious
- Clear separation between normal and anomalous prompt distributions

### Visualizations
1. **PCA Scatter Plots**: Shows true vs predicted labels in 2D space
2. **Decision Boundary**: Mahalanobis distance contours showing detection threshold
3. **Distance Distribution**: Histogram comparing normal vs anomalous prompt distances

## 🚀 Setup Instructions

### Prerequisites
```bash
Python 3.7+
numpy
scipy
matplotlib
scikit-learn
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/spectra-ai-anomaly-detection.git
cd spectra-ai-anomaly-detection

# Install required packages
pip install numpy scipy matplotlib scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

### Running the Notebook
```bash
# Launch Jupyter Notebook
jupyter notebook

# Open: spectra_ai_anomaly_detection.ipynb
# Run all cells sequentially
```

## 📁 Project Structure

```
spectra-ai-anomaly-detection/
│
├── spectra_ai_anomaly_detection.ipynb  # Main notebook
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── anomaly_detection_visualization.png  # Output visualization 1
├── distance_distribution.png            # Output visualization 2
└── report.pdf                          # Project report (if applicable)
```

## 🔍 How It Works

### Data Generation
- Creates 1000 normal prompt embeddings (centered at origin)
- Generates 100 anomalous prompts (shifted mean, different covariance)
- Embedding dimension: 50D (simulating real language model embeddings)

### Detection Pipeline
1. **Training Phase**: Compute mean and covariance from normal prompts only
2. **Distance Calculation**: Measure Mahalanobis distance for all prompts
3. **Statistical Testing**: Apply chi-square test with α = 0.01
4. **Bayesian Update**: Calculate posterior probability for flagged prompts
5. **Visualization**: Project to 2D using PCA and plot results

## 📈 Mathematical Foundation

### Mahalanobis Distance
Measures the distance from a point to a distribution's center, normalized by the distribution's shape:
```
D_M(x) = √((x - μ)ᵀ Σ⁻¹ (x - μ))
```

### Chi-Square Distribution
For multivariate normal data: `D_M²(x) ~ χ²(k)` where k = embedding dimension

### Bayes' Theorem
```
P(Malicious | Flagged) = [P(Flagged | Malicious) × P(Malicious)] / P(Flagged)
```

## 🛡️ Security Implications

### Strengths
- ✅ High detection accuracy with low false positive rate
- ✅ Probabilistic framework provides confidence levels
- ✅ Adapts to the natural distribution of normal prompts
- ✅ No training labels required for normal prompts

### Limitations & Risks
- ⚠️ Assumes normal prompts follow multivariate normal distribution
- ⚠️ Sophisticated adversarial prompts might mimic normal distribution
- ⚠️ Requires sufficient normal data for accurate covariance estimation
- ⚠️ High-dimensional embeddings may suffer from curse of dimensionality

### Bypass Scenarios
If anomalous prompts bypass detection:
- **Injection Attacks**: Malicious instructions executed by the model
- **Data Exfiltration**: Sensitive information extracted through crafted prompts
- **Model Manipulation**: System behavior altered in unexpected ways
- **Reputation Damage**: Harmful outputs attributed to the organization

## 🔮 Future Enhancements

1. **Real Embeddings**: Use actual language model embeddings (BERT, GPT)
2. **Online Learning**: Update distribution as new prompts arrive
3. **Ensemble Methods**: Combine multiple anomaly detection techniques
4. **Adversarial Testing**: Evaluate robustness against evasion attacks
5. **Multi-Modal Detection**: Incorporate semantic analysis alongside statistical methods

## 📚 References

- Mahalanobis Distance: [Wikipedia](https://en.wikipedia.org/wiki/Mahalanobis_distance)
- Chi-Square Distribution: [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
- PCA: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)

## 👥 Author

**KHUSHIN VYAS 22070126056**  
Internship Program - Spectra AI Mini Challenge  
Submission Date: October 27, 2025

---

**Note**: This is a proof-of-concept prototype using synthetic data. For production deployment, use real prompt embeddings and conduct thorough security testing.
