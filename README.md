Polynomial regression (degree 2) in Python using NumPy, Scikit-Learn, and Matplotlib. Generates noisy synthetic quadratic data, applies PolynomialFeatures transformation, trains a LinearRegression model, evaluates test MSE, and visualizes the fitted curve   

# Polynomial Regression on Synthetic Quadratic Data

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Generation-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Polynomial%20Regression-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)

A concise machine learning notebook built with Python, NumPy, Scikit-Learn, and Matplotlib that demonstrates **Polynomial Regression (Degree $2$)** on noisy synthetic data. It walks through reproducible data generation, train-test splitting, polynomial feature expansion, ordinary least squares fitting, Mean Squared Error (`MSE`) evaluation, and curve visualization.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Synthetic Dataset & Mathematical Model](#-synthetic-dataset--mathematical-model)
- [Implementation Workflow](#-implementation-workflow)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Getting Started](#-getting-started)

---

## 🎯 Project Overview

Standard linear models cannot capture non-linear curvature in raw single-feature data. This notebook demonstrates how to model a quadratic relationship using linear regression by transforming the input feature space:
1. **Generates reproducible synthetic data** following a noisy quadratic curve.
2. **Splits the dataset** into training ($80\%$) and testing ($20\%$) subsets.
3. **Expands input features** to degree $2$ using Scikit-Learn's `PolynomialFeatures`.
4. **Trains a `LinearRegression` estimator** on the transformed polynomial features.
5. **Evaluates generalization error** using Mean Squared Error (`MSE`) and visualizes the resulting regression curve.

---

## 📐 Synthetic Dataset & Mathematical Model

Using `np.random.seed(1)` for reproducibility, $N = 100$ data points are generated as follows:

- **Input Feature ($x$):** $100$ samples drawn uniformly from the interval $[0, 2)$ via `2 * np.random.rand(100, 1)`.
- **Target Variable ($y$):** Generated from a second-degree polynomial equation perturbed by standard Gaussian noise:

$$y = 4 + 3x + 1.5x^2 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

- **Train-Test Split:** Partitioned via `train_test_split(..., test_size=0.2, random_state=42)`:
  - **Training Set (`x_train`, `y_train`):** $80$ samples (`(80, 1)`)
  - **Test Set (`x_test`, `y_test`):** $20$ samples (`(20, 1)`)

---

## ⚙️ Implementation Workflow

1. **Polynomial Feature Transformation:**
   `PolynomialFeatures(degree=2, include_bias=False)` maps each scalar input $x$ into a two-dimensional feature vector $[x, x^2]$.
   - `fit_transform` is applied strictly to `x_train` to avoid data leakage.
   - `transform` is applied to `x_test`.
2. **Model Fitting:**
   `LinearRegression()` fits the linear coefficients $\hat{\beta}_1, \hat{\beta}_2$ and intercept $\hat{\beta}_0$ to approximate:

$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2$$

3. **Visualization:**
   - Plots the raw synthetic observations $(x, y)$ using `plt.scatter(x, y)`.
   - Evaluates the trained model over $100$ evenly spaced points across $x \in [0, 2]$ (`np.linspace(0, 2, 100)`) and plots the fitted degree-$2$ polynomial curve.

---

## 📈 Model Evaluation & Results

| Metric / Parameter | Value |
| :--- | :---: |
| **Polynomial Degree** | $2$ |
| **Total Samples** | $100$ ($80$ train / $20$ test) |
| **Noise Variance ($\sigma^2$)** | $1.0$ |
| **Test Mean Squared Error (`MSE`)** | **`0.7840`** (`0.784005`) |

*Because the true data-generating process contains irreducible Gaussian noise with variance $\sigma^2 = 1.0$, a test `MSE` of `0.7840` indicates that the degree-$2$ polynomial model accurately captures the underlying quadratic trend without overfitting.*

---

## 🚀 Getting Started

### Prerequisites
Install the required Python packages:
```bash
pip install numpy matplotlib scikit-learn jupyter
```

### Running the Notebook
1. Clone this repository or download the `.ipynb` notebook file.
2. Launch Jupyter Notebook or open the file in Google Colab:
   ```bash
   jupyter notebook
   ```
3. Execute all cells sequentially to generate the dataset, fit the polynomial regression model, compute the test `MSE`, and render the plots.
