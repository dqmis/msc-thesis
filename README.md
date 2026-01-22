# Equity by Design: Fairness-Driven Recommendation in Heterogeneous Two-Sided Markets

Recommendation systems typically optimize for consumer relevance, but this can unintentionally disadvantage niche users or small producers. This work formalizes consumer utility (relevance) and producer utility (exposure), and introduces Conditional Value at Risk (CVaR) as an optimization objective to directly target fairness.

The project investigates fairness in multi-stakeholder marketplaces (e.g., platforms with both consumers and producers) and proposes optimization methods that balance consumer satisfaction with equitable producer exposure. It integrates machine learning recommender models with fairness-aware allocation strategies and evaluates their performance on real-world and synthetic datasets.


### Key Contributions:

- **Defines consumer and producer utility functions** in two-sided marketplaces
- **Introduces fairness-aware optimization** (mean utility, max–min fairness, CVaR)
- **Demonstrates that fairness constraints can improve business outcomes** such as Sell-Through Rate (STR) and Gross Merchandise Value (GMV)
- **Compares exact solvers** (SCIP, Gurobi) with scalable approximations (relaxation, Augmented Lagrangian)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

1. **Install uv** (if not already installed):

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup the repository**:

```bash
git clone <your-repo-url>
uv sync
```

3. **Verify installation**:

```bash
uv run python -c "import cvxpy, torch, numpy; print('Installation successful!')"
```

## 📁 Repository Structure

```
├── src/                          # Core implementation
│   ├── problems/                 # Optimization problems (CVaR, mean, min)
│   ├── optimizers/              # Solver implementations
│   ├── models/                  # ML model definitions
│   └── measures.py              # Fairness and utility metrics
├── notebooks/                   # Jupyter notebooks for experiments
│   ├── movielens/              # MovieLens dataset experiments
│   ├── experiments/            # Additional experimental results
│   └── *.ipynb                 # Individual analysis notebooks
├── data/                       # Datasets and preprocessed data
│   ├── ml-100k/               # MovieLens 100K dataset
│   ├── *_predictions.npy      # Model predictions
│   └── *_user_groups.json     # User group assignments
├── outputs/                    # Output files (logs, results)
└── results/                    # Experimental results and visualizations

```

## 🔬 Reproducing Results

### 1. Basic Fairness Analysis

Start with the main experiment notebook:

```bash
uv run jupyter notebook notebooks/movielens/experiments.ipynb
```

This notebook demonstrates:

- Loading and preprocessing the MovieLens dataset
- Training recommendation models
- Running fairness-aware optimization experiments
- Generating tradeoff curves between consumer utility and producer fairness

### 2. CVaR Fairness Experiments

For Conditional Value at Risk experiments:

```bash
uv run jupyter notebook notebooks/movielens/cvar.ipynb
```

### 3. Custom Experiments

To run your own experiments, use the core optimization functions:

```python
from src.problems.problems import compute_consumer_optimal_solution
import numpy as np

# Example: Run CVaR optimization
rel_matrix = np.random.rand(100, 50)  # 100 users, 50 items
k_rec = 10  # Recommend 10 items per user
gamma = 0.8  # Fairness constraint strength
group_assignments = [0, 1, 0, 1, ...]  # User group assignments
alpha = 0.1  # CVaR confidence level

# Solve the optimization problem
optimal_value, allocations = compute_consumer_optimal_solution(
    rel_matrix=rel_matrix,
    k_rec=k_rec,
    producer_max_min_utility=5.0,
    gamma=gamma,
    method="cvar",
    group_assignments=group_assignments,
    alpha=alpha
)
```

## 📊 Key Results

The experiments demonstrate several important findings:

1. **Fairness-Utility Tradeoffs**: There exists a clear tradeoff between consumer utility and producer fairness
2. **CVaR Effectiveness**: CVaR-based optimization provides better worst-case guarantees than mean utility optimization
3. **Business Impact**: Fairness constraints can improve long-term business metrics like STR and GMV
4. **Scalability**: Relaxed optimization methods provide good approximations with significantly reduced computational cost

## 🛠️ Available Optimization Methods

The repository implements several fairness-aware optimization approaches:

- **`mean`**: Maximizes average consumer utility
- **`min`**: Maximizes minimum consumer utility (max-min fairness)
- **`cvar`**: Minimizes Conditional Value at Risk for group fairness
- **`cvar_relaxed_naive`**: CVaR with naive rounding
- **`cvar_relaxed_topk`**: CVaR with top-k rounding

## 📈 Datasets

The experiments use several datasets:

- **MovieLens 100K**: Standard collaborative filtering benchmark
- **Amazon Reviews**: Large-scale e-commerce data
- **Synthetic Data**: Controlled experiments for validation

## 🔧 Dependencies

Key dependencies include:

- **CVXPY**: Convex optimization framework
- **Gurobi/SCIP**: Commercial/open-source solvers
- **PyTorch**: Deep learning models
- **NumPy/SciPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization

See `pyproject.toml` for the complete dependency list.

### Common Issues:

1. **Solver not found**: Make sure you have Gurobi or SCIP installed and properly licensed
2. **Memory issues**: For large datasets, consider using the relaxed optimization methods
3. **Installation problems**: Ensure you're using Python 3.12+ and the latest version of uv

### Getting Help:

- Check the Jupyter notebooks for usage examples
- Review the docstrings in `src/problems/problems.py` for detailed function documentation
- Open an issue for specific problems or questions
