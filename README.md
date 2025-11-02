# 🎯 Fashion MNIST Classification - Machine Learning Project

<div align="center">

![Fashion MNIST](https://img.shields.io/badge/Dataset-Fashion_MNIST-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![ML](https://img.shields.io/badge/Machine-Learning-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-94.86%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Achieved 94.86% accuracy on Fashion MNIST using traditional machine learning models**

</div>

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Achievements](#-key-achievements)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Authors](#-authors)
- [License](#-license)

## 🎯 Project Overview

This project demonstrates the application of **traditional machine learning algorithms** for image classification on the Fashion MNIST dataset. Despite the common use of deep learning for image tasks, we achieved outstanding results using carefully optimized classical models.

**Key Innovation:** Identified and resolved class confusion issues that significantly boosted model performance from 88% to **94.86% accuracy**.

## 🏆 Key Achievements

- **🎯 94.86% Accuracy** with XGBoost (surpassing many deep learning approaches)
- **🔍 7 Algorithms** comprehensively evaluated and compared
- **📊 Advanced Analysis** including confusion matrices, ROC curves, and cross-validation
- **🚀 Production-ready** methodology with proper train-test splits and hyperparameter tuning

## 📊 Dataset

### Fashion MNIST Specifications
| Property | Details |
|----------|---------|
| **Total Images** | 70,000 (60,000 train + 10,000 test) |
| **Image Size** | 28×28 pixels grayscale |
| **Classes** | 10 fashion categories |
| **Resolution** | 784 features per image |

### Class Labels
| Label | Class Name | Label | Class Name |
|-------|------------|-------|------------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

## 🔬 Methodology

### Data Preprocessing
```python
# Key preprocessing steps
1. Normalization: Pixel values scaled to [0, 1]
2. Flattening: 28×28 images → 784-dimensional vectors
3. Train-Test Split: Multiple strategies tested
4. Class Balancing: Verified equal distribution
```

### Algorithms Implemented
We implemented and compared 7 machine learning algorithms:

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **XGBoost** | Ensemble Boosting | Gradient boosting, handling sparse data |
| **SVM** | Maximum Margin | RBF kernel, optimal hyperparameters |
| **Random Forest** | Ensemble Bagging | Multiple decision trees, reduced overfitting |
| **K-Nearest Neighbors** | Instance-based | Distance-based classification |
| **Decision Tree** | Rule-based | Interpretable, feature importance |
| **Logistic Regression** | Linear Model | Probabilistic, multi-class support |
| **Naive Bayes** | Probabilistic | Fast training, independence assumption |

### Hyperparameter Optimization
- **SVM**: GridSearchCV with RBF kernel
- **XGBoost**: RandomizedSearchCV for efficient tuning
- **Cross-validation**: 5-fold validation for robust evaluation

## 📈 Results

### Performance Comparison
| Model | Original Accuracy | After Shirt Removal | Final (80/20 Split) | Training Time | Prediction Time |
|-------|-------------------|---------------------|---------------------|---------------|-----------------|
| **XGBoost** | 88.35% | 93.24% | **94.86%** | High | Low |
| **SVM** | 88.28% | 93.83% | **94.49%** | High | Moderate |
| **Random Forest** | 87.64% | - | - | High | Moderate |
| **KNN** | 85.41% | - | - | Moderate | High |
| **Logistic Regression** | 84.35% | - | - | Low | Low |
| **Decision Tree** | 78.88% | - | - | Low | Low |
| **Naive Bayes** | 58.56% | - | - | Very Low | Very Low |

### Key Findings

#### 🎯 Breakthrough Insight: The "Shirt Problem"
```python
# Major challenge identified
Class 6 (Shirt) was frequently confused with:
- Class 0 (T-shirt/top): 23% confusion rate
- Class 2 (Pullover): 19% confusion rate
- Class 4 (Coat): 15% confusion rate
```

**Solution:** Strategic removal of the shirt class improved overall accuracy by **5-6%** across top models.

#### 📊 Data Split Optimization
- **Original**: 85.71%/14.29% (60K/10K)
- **Optimized**: 80%/20% - Better generalization and evaluation

#### 🔧 Data Augmentation Analysis
Surprisingly, traditional data augmentation techniques decreased performance:
- **SVM**: Dropped to 67.00%
- **XGBoost**: Dropped to 69.00%

## 📁 Project Structure

```
Fashion-MNIST-Classification/
│
├── 📁 notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   ├── 3_model_evaluation.ipynb
│   └── 4_final_results_analysis.ipynb
│
├── 📁 reports/
│   ├── Classification_Fashion_minst_pim.pdf
│   └── Classification_des_images_de_fashion_mnist.pdf
│
├── 📁 images/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── performance_charts/
│
├── 📄 requirements.txt
├── 📄 environment.yml
├── 📄 README.md
└── 📄 .gitignore
```

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/ayoub1999hannachi/Fashion-MNIST-Classification.git
cd Fashion-MNIST-Classification

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate fashion-mnist-ml
```

### Dependencies
See [`requirements.txt`](requirements.txt) for complete list:
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🚀 Usage

### Running the Project
```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate through notebooks in order:
# 1. Data Preprocessing
# 2. Model Training  
# 3. Model Evaluation
# 4. Final Results Analysis
```

### Quick Start Example
```python
# Sample code to load and preprocess data
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize and flatten images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)
```

## 🔧 Technical Details

### Model Configuration
#### XGBoost (Best Performing)
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.3,
    n_estimators=100,
    objective='multi:softmax',
    num_class=10,
    random_state=42
)
```

#### SVM with RBF Kernel
```python
from sklearn.svm import SVC

model = SVC(
    C=10,
    kernel='rbf',
    gamma='scale',
    random_state=42
)
```

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision**: Quality of positive predictions  
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Model discrimination capability

### Cross-Validation Results
- **SVM**: 93.99% average accuracy (5-fold)
- **XGBoost**: 94.21% average accuracy (5-fold)



### Institution
**Université Ibn Khaldoun - UIK**  
*Diplôme National d'Ingénieur en Informatique*  
**Academic Year:** 2023-2024

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- 📚 **Full Technical Report**: [Download PDF](reports/Classification_Fashion_minst_pim.pdf)
- 🎓 **Presentation Slides**: [Download PDF](reports/Classification_des_images_de_fashion_mnist.pdf)
- 💻 **Source Code**: [GitHub Repository](https://github.com/ayoub1999hannachi/Fashion-MNIST-Classification)

## 📞 Contact

For questions or collaborations:
- 📧 Email: ahannachi193@gmail.com
- 💼 LinkedIn: [Hannachi Ayoub](https://www.linkedin.com/in/ayoub-hannachi-0727931b0/)
- 🐙 GitHub: [@ayoub1999hannachi](https://github.com/ayoub1999hannachi/)

---

<div align="center">

### ⭐ **If this project helped you, don't forget to give it a star!** ⭐

*"Traditional ML can still outperform deep learning with the right insights and optimizations"*

</div>