# Spaceship Titanic Prediction Project
## Overview
This project aims to predict whether passengers aboard the Spaceship Titanic were transported to another dimension following a cosmic anomaly. Using a dataset from a Kaggle competition, I implemented machine learning and deep learning models to classify passengers based on various features such as age, home planet, and spending on amenities. The project involves data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and submission preparation.

<img src="eda&results/precision, recall, f1_score.png> 

## Project Structure
**Data:** 
* `train.csv`: Training dataset with passenger details and the target variable  `Transported`.
* `test.csv`: Test dataset for generating predictions.

**Notebook:** `titanic-spaceship.ipynb` contains the complete implementation.
**Output:** `submission.csv` contains predictions for the test dataset.
**Dependencies:** Python libraries including pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, and keras-tuner.

## Implementation Steps
**1. Data Loading and Exploration**
**Objective**: Understand the dataset structure and identify missing values.
**Actions**:
* Loaded `train.csv` and `test.csv` using pandas.
* Displayed the first few rows of both datasets to inspect features.
* Used `.info()` to check data types and non-null counts.
* Calculated missing values with `.isnull().sum()`.

**Findings:**
* Training data: 8,693 entries, 14 columns (13 features + `Transported`).
* Test data: 4,277 entries, 13 columns (no `Transported`).
* Missing values were present in most columns, notably `HomePlanet`, `CryoSleep`, `Cabin`, `Age`, and spending-related columns.

## 2. Data Preprocessing
**Objective**: Clean and prepare data for modeling.
**Actions**:
**Missing Values:**
* Filled `Age` with median values.
* Filled `HomePlanet`, `Destination`, `Cabin`, and `VIP` with mode values.
* Set `CryoSleep` missing values to `False`.
* Filled spending columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) with 0.

**Feature Encoding:**
* Used `LabelEncoder` for categorical columns (`HomePlanet`, `Destination`, `Cabin`).

**Feature Scaling:**
* Applied `StandardScaler` to normalize numerical features for deep learning models.

**Data Splitting:**
* Split training data into `x_train`,`x_test`, `y_train`, and `y_test` using `train_test_split`.

**Outcome**: All missing values were handled except for `Name` (not used in modeling). Data was encoded and scaled for model compatibility.

## 3. Exploratory Data Analysis (EDA)
**Objective**: Identify patterns and outliers in numerical features.
**Actions:**
* Selected numerical columns (`Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`).
* Created boxplots using seaborn to visualize distributions and outliers.
**Findings**:
* Spending columns showed significant outliers, indicating varied passenger spending behaviors.
* `Age` had a relatively normal distribution with some outliers.

Visual Results: (Add boxplot visualizations here)

## 4. Model Training
**Objective**: Train and evaluate multiple models to predict `Transported`.
**Models**:
* Traditional Machine Learning:
* Decision Tree Classifier
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

**Deep Learning:**
Neural network using TensorFlow/Keras.

**Actions**:
**Traditional ML:**
* Performed 5-fold cross-validation on all models using `cross_val_score`.
* Random Forest Classifier achieved the highest mean accuracy (~0.793).

**Deep Learning:**
* Built a Sequential model with two hidden layers (64 and 32 units, ReLU activation), dropout (0.3), and a sigmoid output layer.
* Compiled with `adam` optimizer and `binary_crossentropy` loss.
* Used `EarlyStopping` to prevent overfitting (patience=5).
* Trained for 50 epochs with a batch size of 32 and 20% validation split.
* Evaluated on test set, achieving ~0.73 accuracy.

## 5. Hyperparameter Tuning
**Objective**: Optimize the deep learning model for better performance.
**Actions**:
* Used `keras-tuner` (RandomSearch) to tune:
* Number of units in hidden layers (32–128 for layer 1, 16–64 for layer 2).
* Dropout rates (0.2–0.5).
* Optimizer (`adam`, `rmsprop`, `sgd`).

Ran 10 trials with `EarlyStopping`.
Best hyperparameters:
* Units: 64 (layer 1), 64 (layer 2).
* Dropout: 0.3 (both layers).
* Optimizer: `adam`.

Trained final model with best hyperparameters, achieving ~0.73 test accuracy.

**Outcome**: Tuning slightly improved validation accuracy (~0.77), but test accuracy remained comparable to the initial model.

## 6. Prediction and Submission
**Objective**: Generate predictions for the test dataset and prepare a submission file.
**Actions**:
* Used the final tuned model to predict `Transported` for `test_data`.
* Converted predictions to boolean values.
* Created `submission.csv` with `PassengerId` and `Transported` columns.

**Outcome**: Submission file was successfully generated and saved.

## Results
**Traditional Machine Learning:**
* Random Forest Classifier performed best with a cross-validation accuracy of ~79.3%.
* Other models (Decision Tree, Logistic Regression, Gradient Boosting) achieved accuracies between 79.0% and 79.3%.

**Deep Learning:**
* Initial neural network: ~73% test accuracy.
* Tuned neural network: ~73% test accuracy, with a peak validation accuracy of ~77%.

**Key Observations:**
* Traditional ML models outperformed the deep learning model, likely due to the dataset’s moderate size and feature complexity.
* The deep learning model showed potential but required extensive tuning to match traditional models.
* Outliers in spending columns may have influenced model performance, suggesting potential for feature engineering (e.g., log transformation).

Visual Results: (Add training/validation loss and accuracy plots from history here)

## How to Run
**1. Prerequisites:**
- Install required libraries:
  ```bash:
  pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner
  ```
- Ensure Python 3.10+ and a GPU-enabled environment (if using Kaggle or similar).

**2. Steps:**
- Clone the repository or download `titanic-spaceship.ipynb`.
- Place `train.csv` and `test.csv` in the working directory.
- Run the notebook cells sequentially.
- The final `submission.csv` will be generated in the working directory.

## Future Improvements
**1. Feature Engineering:**
Extract deck, number, and side from `Cabin` for more granular features.
Apply log transformation to spending columns to reduce outlier impact.

**2. Model Enhancements:**
Experiment with ensemble methods combining Random Forest and Gradient Boosting.
Explore advanced neural network architectures (e.g., batch normalization, additional layers).

**3. Data Augmentation:**
Address missing values with more sophisticated imputation techniques (e.g., KNN imputation).

**4. Evaluation:**
Use additional metrics (precision, recall, F1-score) to evaluate model performance comprehensively.

## Conclusion
The project successfully implemented a pipeline for predicting passenger transportation in the Spaceship Titanic dataset. Random Forest achieved the highest accuracy (~79.3%), while the tuned deep learning model reached ~73% test accuracy. The results highlight the effectiveness of traditional ML for this dataset, with opportunities for further improvement through feature engineering and model optimization.
