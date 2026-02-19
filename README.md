Here is a complete, well-structured `README.md` file for your repository based on the provided script and the implied project structure.

---

# Piraeus Vice: Crime Classification & Killer Identification

## 📌 Project Overview

**Piraeus Vice** is a comprehensive machine learning and data analysis project aimed at identifying perpetrators (`killer_id`) based on various crime scene characteristics. The project involves exploratory data analysis (EDA), density estimation, dimensionality reduction, and the application of both supervised and unsupervised machine learning models to classify and predict the responsible criminal.

The codebase systematically addresses a series of 8 analytical tasks (Q1-Q8), ranging from basic data exploration and covariance analysis to complex neural networks and clustering.

## 📂 Repository Structure

```text
PiraeusVice/
│
├── crimes.csv                       # The main dataset (Train/Val/Test splits)
├── solutions_Q1_Q8.py               # Main Python script with data processing and modeling
│
├── outputs/                         # Generated visualizations and results
│   ├── outputsq1_*.png              # Histograms, GMM fit, 2D scatter plots
│   ├── outputsq2_*.png              # Covariance heatmaps, 95% confidence ellipses
│   ├── outputsq3_*.png              # Bayes decision regions, confusion matrix
│   ├── outputsq4_*.png              # Logistic Regression comparisons and matrices
│   ├── outputsq5_*.png              # SVM decision boundaries and SVs
│   ├── outputsq6_*.png              # MLP confusion matrix, Permutation Feature Importance
│   ├── outputsq7_*.png              # PCA Scree plot, variance analysis, 2D PCA scatter
│   ├── outputsq8_*.png              # K-Means clustering results
│   ├── model_comparison.png         # Final validation accuracy comparison chart
│   └── outputssubmission.csv        # Final model predictions on the dataset
│
├── presentations/                   # Project documentation
│   ├── piraeus_vice_report.pdf      # Detailed project report
│   └── piraeus_vice_slides.pdf      # Presentation slides
│
└── .idea/                           # IDE configuration files

```

## 📊 Dataset Features

The model utilizes a mixture of continuous and categorical variables from `crimes.csv`:

* **Continuous Features:** `hour_float`, `latitude`, `longitude`, `victim_age`, `temp_c`, `humidity`, `dist_precinct_km`, `pop_density`
* **Categorical Features:** `weapon_code`, `scene_type`, `weather`, `vic_gender` (Processed using One-Hot Encoding)
* **Target Variable:** `killer_id`

## 🚀 Analytical Workflow & Models

The `solutions_Q1_Q8.py` script executes the following stages sequentially:

* **Q1 | Data Exploration & Density Estimation:** Generates feature histograms and fits a 3-component Gaussian Mixture Model (GMM) to temporal data (`hour_float`).
* **Q2 | Covariance & Confidence Ellipses:** Computes manual and library-based log-likelihoods, generating correlation heatmaps and plotting 95% confidence ellipses for geographic distributions (latitude vs. longitude).
* **Q3 | Bayesian Classification:** Implements a custom multivariate Gaussian Naive Bayes classifier utilizing Mahalanobis distance. Visualizes decision boundaries using PCA reduction.
* **Q4 | Linear Classification:** Applies Logistic Regression (LBFGS solver) for linear baseline comparison against the Bayes approach.
* **Q5 | Non-Linear Support Vector Machine:** Trains an SVM with an RBF kernel, visualizing the non-linear decision boundaries and Support Vectors in 2D PCA space.
* **Q6 | Neural Networks & Feature Importance:** Trains a Multi-Layer Perceptron (MLP) Classifier (128-64-32 architecture). Calculates and plots permutation feature importance to identify the main drivers of the prediction.
* **Q7 | Dimensionality Reduction (PCA):** Analyzes eigenvalues and cumulative variance via Scree plots, establishing the optimal number of principal components to retain 90% of the variance.
* **Q8 | Unsupervised Learning (K-Means):** Applies K-Means clustering to the PCA-reduced space, mapping the resulting clusters back to the actual killer identities to evaluate unsupervised accuracy.

Finally, the script evaluates the validation accuracy of all supervised models and the clustering model, outputting a unified `model_comparison.png` and generating the `submission.csv` using the highest-performing MLP predictions.

## 🛠 Dependencies

To run the code, you will need Python 3.x and the following libraries:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scipy`
* `scikit-learn`

Install the dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn

```

## ⚙️ How to Run

1. Clone the repository.
2. Ensure the `crimes.csv` dataset is present in the root directory or adjust the `DATA` and `OUT` path variables in the python script.
3. Execute the main script:

```bash
python solutions_Q1_Q8.py

```

4. Check the `/outputs` directory for all generated charts and the final `submission.csv` prediction file.
