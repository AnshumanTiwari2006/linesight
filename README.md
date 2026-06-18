# LineSight 🔍

**LineSight** is an educational, "glass-box" machine learning library built for understanding—not just executing—regression models. 

Where traditional libraries like `scikit-learn` act as black boxes (you put data in, you get predictions out), LineSight is explicitly designed to teach you **how** the algorithm works under the hood. Every model can visually show you its loss surface, animate its gradient descent path, and explain its mathematical equations in plain English.

---

## 🏗️ Library Architecture

LineSight is built on a strict, pedagogical tri-layer design. Every single model in the library is broken down into three distinct modules:

1. **`engine` (The Math)**
   The core implementation using pure NumPy. Instead of opaque, heavily-optimized C-extensions, the engines are written in highly readable Python. We use explicit `weights` and `bias` variables instead of unified `theta` vectors so the code reads exactly like a textbook formula.
   
2. **`explain` (The Interpretation)**
   A suite of functions that translate the mathematical state of the model into plain-English sentences. It explains the effect of the L2 penalty, parses out the slope coefficient, and dynamically tells you what the model learned.

3. **`visualization` (The Proof)**
   A massive collection of Matplotlib-based functions. You can visualize the dataset, the residuals, the 3D loss surface, and even generate live animations of the algorithm learning step-by-step.

---

## 🧠 The Models

LineSight currently implements 6 foundational regression models, each designed to teach a specific mathematical concept:

| Model | Algorithm | What it teaches |
|---|---|---|
| `LinearRegression` | Gradient Descent | The baseline concept of minimizing Mean Squared Error (MSE). |
| `MultipleLinearRegression` | Gradient Descent | Dealing with planes, hyperplanes, and multicollinearity. |
| `PolynomialRegression` | Gradient Descent | Basis expansion and the bias-variance tradeoff (underfitting vs overfitting). |
| `RidgeRegression` | Gradient Descent | L2 Regularization (weight decay) to handle correlated features. |
| `LassoRegression` | Coordinate Descent | L1 Regularization and automatic feature selection (sparsity). |
| `ElasticNetRegression` | Coordinate Descent | Blending L1 and L2 penalties via the `l1_ratio`. |
| `LogisticRegression` | Gradient Descent | Binary classification, the Sigmoid function, and Log Loss. |

---

## ⚡ The Unified API

All models strictly follow a unified, `scikit-learn`-style API, meaning you already know how to use them. 

### 1. Core Execution
```python
model = RidgeRegression(alpha=0.1, learning_rate=0.01)
model.fit(X, y)                    # Trains the model
predictions = model.predict(X)     # Generates predictions
metrics = model.score(X, y)        # Returns {'r2', 'rmse', 'mae', 'mse'}
```

### 2. Context Panels & Explanations
Every visualization and summary natively includes **Context Panels**—live, dynamically generated text boxes attached directly to the plots that explain the theory, the formula, and how to read the chart.
```python
model.summary()                    # Prints training summary and final metrics
model.show_equation()              # Prints the literal mathematical formula (e.g., y = 3.5x + 1.2)
model.explain_regularization()     # Explains the specific L1/L2 penalty active in the model
```

### 3. Visualizations & Animations
Instead of just returning predictions, LineSight lets you look inside the optimization process:
```python
# Visualize the exact distance (error) of every point from the line
model.plot_residuals(X, y)

# Watch the line physically shift and tilt as it minimizes loss over epochs
model.animate_training(X, y)

# See the 3D bowl of the loss surface and the path the gradient took
model.plot_loss_surface(X, y)

# (Lasso/Ridge) Watch features get eliminated as the penalty increases
model.plot_coefficient_shrinkage(X, y)
```

---

## 🚀 Quick Start

### Installation

For local usage and development:
```bash
git clone https://github.com/AnshumanTiwari2006/linesight
cd linesight
pip install -e ".[dev]"
```

### Example Usage

```python
import numpy as np
from linesight import LinearRegression

# 1. Generate some dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.0, 5.8, 8.1, 9.9])

# 2. Instantiate and train the model (store_history is required for animations)
model = LinearRegression(learning_rate=0.01, epochs=1000, store_history=True)
model.fit(X, y)

# 3. Understand what the model learned
model.show_equation()          # Prints: y = 1.9800x + 0.1200
model.explain_coefficients()   # Explains what 1.98 means in plain English

# 4. Prove it visually
model.plot_fit(X, y)           # View the regression line
model.plot_loss_curve()        # Ensure the learning rate was stable
```

---

## 🧪 Running Tests

LineSight is heavily tested to ensure mathematical correctness matches `scikit-learn` implementations. To run the test suite:

```bash
pytest tests/ -v
```
*(109 tests covering validators, all model engines, training history, metrics, and explanation layers).*

---

## 📄 License

MIT License. Built for education, transparency, and the love of math.
