# ELECTRIC VEHICLE PRICE PREDICTION USING SUPPORT VECTOR MACHINES(SVM)


![electric vehicle](https://images.unsplash.com/photo-1613401381243-6a97cf70f0b7)

Image source: Unsplash

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Objective](#objective)
- [Project Steps](#project-steps)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Handling Duplicates](#handling-duplicates)
  - [Outlier Detection and Treatment](#outlier-detection-and-treatment)
  - [Encoding Categorical Features](#encoding-categorical-features)
- [Modeling](#modeling)
  - [Train-Test Split](#train-test-split)
  - [Model Selection](#model-selection)
  - [Model Performance Evaluation](#model-performance-evaluation)
  - [Model Improvement Techniques](#model-improvement-techniques)
    - [Feature Selection](#feature-selection)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Trying Different Algorithms](#trying-different-algorithms)
- [How to Run the Project](#how-to-run-the-project)
- [Real-World Applications](#real-world-applications)
- [Visualizations and Model Comparison](#visualizations-and-model-comparison)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

---

## Project Overview
This project aims to predict the price of electric vehicles using the Support Vector Machine (SVM) algorithm. The dataset contains information on electric vehicles registered with the Washington State Department of Licensing (DOL). The task is to predict the vehicle's price based on various features, such as the vehicle's make, model, electric range, and base MSRP (Manufacturer's Suggested Retail Price).

## Dataset Description
The dataset used for this project consists of Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs). The data includes vehicle specifications, such as electric range, model year, and base MSRP.

### Dataset Columns:
- **VIN (1-10)**: First 10 characters of the Vehicle Identification Number (VIN)
- **County**: The county where the vehicle owner resides
- **City**: City of the registered vehicle
- **State**: State of the registered vehicle
- **ZIP Code**: Zip code of the vehicle owner
- **Model Year**: The vehicle's model year
- **Make**: Manufacturer of the vehicle
- **Model**: Vehicle model
- **Electric Vehicle Type**: All-electric or plug-in hybrid
- **CAFV Eligibility**: Clean Alternative Fuel Vehicle eligibility status
- **Electric Range**: The vehicle's electric-only range in miles
- **Base MSRP**: The vehicle's base MSRP
- **Expected Price**: The predicted or expected price of the vehicle

### Dataset Source:
Kaggle competition: Electric Vehicle Price Prediction  
[Dataset Link](https://drive.google.com/file/d/1kZ299dY3rKLvvnfTsaPtfrUbZb7k31of/view)

---

## Objective
The primary goal of this project is to predict the expected price of electric vehicles based on various features like electric range, model, make, and base MSRP.

## Project Steps
1. **Data Importation & Exploration**: Load and understand the dataset.
2. **Data Preprocessing**: Handle missing values, duplicates, outliers, and encode categorical variables.
3. **Model Selection & Training**: Train a Support Vector Regression (SVR) model using the preprocessed data.
4. **Model Evaluation**: Evaluate the model based on mean squared error (MSE) and R-squared score.
5. **Model Improvement Techniques**: Experiment with hyperparameter tuning and different algorithms for improved performance.

---

## Data Preprocessing

### Handling Missing Values
Some columns like **Electric Utility** and **Vehicle Location** had missing values, which were filled using the mode (most frequent value) for each column.

### Handling Duplicates
No duplicate rows were found in the dataset after running the `df.duplicated()` check.

### Outlier Detection and Treatment
Outliers were detected in **Base MSRP** using Z-scores. Outliers beyond a Z-score threshold of 3 were removed to improve model performance.

### Encoding Categorical Features
Label Encoding was applied to categorical columns like **Make**, **Model**, **Electric Vehicle Type**, and **CAFV Eligibility** using `LabelEncoder()` from scikit-learn.

---

## Modeling

### Train-Test Split
The dataset was split into an 80% training set and a 20% test set. The features selected were:
- **Electric Range**
- **Base MSRP**
- **Model Year**
- **Make**
- **Electric Vehicle Type**
- **CAFV Eligibility**

The target variable was **Expected Price**.

### Model Selection
A **Support Vector Regression (SVR)** model was chosen for this task. SVR is effective for regression tasks with continuous target variables.

### Model Performance Evaluation
- **Mean Squared Error (MSE)**: 3034.01
- **R-squared (R2) Score**: 0.008

The SVR model's initial performance showed a relatively high MSE, indicating that the predicted prices were far from the actual values. The low R-squared score indicated that the model explained only a small portion of the variance in the target variable.

---

## Model Improvement Techniques

### Feature Selection
Features like **Electric Range**, **Base MSRP**, and **Model Year** were retained for further experimentation, as they are highly relevant for price prediction.

### Hyperparameter Tuning
To improve the SVR model, hyperparameters such as the kernel, regularization parameter (C), and gamma can be tuned using **GridSearchCV** or **RandomizedSearchCV**.

### Trying Different Algorithms
In addition to SVR, experimenting with other models like **Random Forest Regressor**, **Gradient Boosting Regressor**, and **XGBoost** may yield better performance.

---

## How to Run the Project
1. Clone the repository and install dependencies.
2. Download the dataset from the provided link.
3. Run the notebook to train the SVM model and evaluate its performance.

---

## Real-World Applications
- **Vehicle Pricing Models**: Predicting vehicle prices helps consumers, dealers, and manufacturers make informed decisions about electric vehicles.
- **Electric Vehicle Valuation**: Automated systems can use these predictions for valuations in insurance or financing applications.

---

## Visualizations and Model Comparison

### Scatter Plot for Base MSRP:
![Base MSRP scatter plot](https://github.com/cliffordnwanna/PREDICTIVE_MODELLING/raw/main/IMAGES/base-msrp-scatter.png)

### Heatmap of Feature Correlations:
![Correlation Heatmap](https://github.com/cliffordnwanna/PREDICTIVE_MODELLING/raw/main/IMAGES/heatmap.png)

---

## Conclusion
- The SVM model initially underperformed in terms of MSE and R-squared score. Further improvement is needed through hyperparameter tuning and experimenting with other algorithms.
- Feature engineering and selection will play a key role in improving prediction accuracy.
- The project has potential applications in electric vehicle pricing models and valuation systems.

---

## Future Work
- Further optimize the SVR model's hyperparameters to improve prediction accuracy.
- Experiment with additional machine learning models like Random Forest or XGBoost.
- Explore neural networks for better regression performance.

---

## Contributing
We welcome contributions to improve this project. To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

---

## License
This project is licensed under the MIT License.

---

## Contact Information
For any inquiries or support, contact:

**Your Name**  
*Email*: [your-email@gmail.com](mailto:your-email@gmail.com)  
*GitHub*: [your-github-profile](https://github.com/your-profile)

---

