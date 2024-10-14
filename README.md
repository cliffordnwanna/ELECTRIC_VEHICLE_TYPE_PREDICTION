

# Electric Vehicle Type Prediction using Support Vector Machines (SVM)

![Feature Importance](https://github.com/cliffordnwanna/ELECTRIC_VEHICLE_TYPE_PREDICTION/raw/main/IMAGES/Electric%20Vehicle.jpg)

## 📋 Project Overview

This project aims to build a predictive model to classify electric vehicles as either **Battery Electric Vehicles (BEVs)** or **Plug-in Hybrid Electric Vehicles (PHEVs)**. Using a dataset of electric vehicles registered across different counties in Washington State, we applied **Support Vector Machines (SVM)** to analyze patterns and predict the type of electric vehicle based on several key features. The goal is to assist in understanding electric vehicle adoption trends and geographical distribution, as well as to provide insights for future electric vehicle-related decisions.

## 📂 Dataset Description

The dataset contains information on electric and plug-in hybrid vehicles, capturing various features such as vehicle make, model, electric range, and location. It is sourced from the Washington State Department of Licensing, providing real-world data on registered electric vehicles. 

**Dataset Source**: [Electric Vehicle Population Data](https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD)

### Features

- **VIN (1-10)**: Partial Vehicle Identification Number
- **County**: The county where the vehicle is registered
- **City**: The city of registration
- **State**: The state of registration (all entries are 'WA')
- **Postal Code**: ZIP code where the vehicle is registered
- **Model Year**: Year when the vehicle model was manufactured
- **Make**: Manufacturer of the vehicle (e.g., Tesla, Nissan, Chevrolet)
- **Model**: Model name of the vehicle
- **Electric Vehicle Type**: Battery Electric Vehicle (BEV) or Plug-in Hybrid Electric Vehicle (PHEV)
- **Clean Alternative Fuel Vehicle (CAFV) Eligibility**: Indicates eligibility for clean alternative fuel incentives
- **Electric Range**: Electric-only range of the vehicle in miles
- **Base MSRP**: Manufacturer's Suggested Retail Price
- **Legislative District**: Legislative district where the vehicle is registered
- **DOL Vehicle ID**: Department of Licensing vehicle ID
- **Vehicle Location**: Geographical coordinates (latitude, longitude)
- **Electric Utility**: Electric utility company serving the area
- **2020 Census Tract**: Census tract for demographic analysis

**Target Column**:
- `Electric Vehicle Type`: The goal is to predict whether a vehicle is a BEV or a PHEV.

### Dataset Link
You can download the dataset [here](https://drive.google.com/file/d/1G-qJedoJuZftjMyV2CbinTzyJPSXpqUO/view?usp=sharing).

## ⚙️ Project Workflow

### 1. Data Preprocessing
- **Feature Selection**: Selected relevant features, dropping unnecessary columns.
- **Handling Missing Values**: Filled missing values with the median (for numerical features) and mode (for categorical features).
- **Encoding**: Encoded categorical features and the target variable using Label Encoding.
- **Scaling**: Scaled numerical features to ensure the SVM model can process the data effectively.

### 2. Model Development
- **Support Vector Machine**: Utilized a linear SVM to classify the electric vehicles into BEV and PHEV.
- **Training & Testing**: Split the dataset into training (80%) and testing (20%) sets for model evaluation.
- **Model Evaluation**: Evaluated the model using accuracy, confusion matrix, and classification report.

### 3. Visualizations
- **Confusion Matrix Heatmap**: Visual representation of model predictions vs actual labels.
- **Feature Importance**: For linear SVMs, identified important features by examining coefficients.

## 📊 Results & Interpretation

### Model Performance

- **Accuracy**: The SVM model achieved an overall accuracy of **86.29%**, indicating it correctly classified 86% of the vehicles.
- **Confusion Matrix**:
    - **True Positives (BEV)**: 31,657 correctly classified as BEVs.
    - **False Positives (BEV)**: 630 instances were incorrectly classified as BEVs but were actually PHEVs.
    - **True Negatives (PHEV)**: 3,798 correctly classified as PHEVs.
    - **False Negatives (PHEV)**: 5,003 instances were incorrectly classified as PHEVs but were actually BEVs.
- **Precision, Recall, and F1-Score**:
    - **BEV**: Precision = 0.86, Recall = 0.98, F1-Score = 0.92
    - **PHEV**: Precision = 0.86, Recall = 0.43, F1-Score = 0.57

### Conclusion:
The SVM model performs well for predicting **BEVs**, with high precision and recall. However, it struggles with accurately identifying **PHEVs**, suggesting possible data imbalance or insufficient feature differentiation. Further enhancements, such as data balancing and additional feature engineering, can improve classification performance.

## 📈 Visualizations

### Confusion Matrix Heatmap
![Confusion Matrix Heatmap](https://github.com/cliffordnwanna/ELECTRIC_VEHICLE_TYPE_PREDICTION/raw/main/IMAGES/confusion_matrix.png)

### Feature Importance (For Linear SVM)
![Feature Importance](https://github.com/cliffordnwanna/ELECTRIC_VEHICLE_TYPE_PREDICTION/raw/main/IMAGES/feature%20selection.png)

The plots provide insights into how the SVM model makes its decisions and which features contribute most to the classification process.

## 🛠️ Recommendations for Improvement

1. **Address Data Imbalance**:
   - Use oversampling (e.g., SMOTE) or undersampling techniques to balance the dataset.
   - Experiment with class weighting in the SVM model to penalize misclassification of PHEVs more heavily.

2. **Feature Engineering**:
   - Consider deriving additional features (e.g., interaction terms, power, efficiency) to enhance differentiation between BEVs and PHEVs.

3. **Parameter Tuning**:
   - Experiment with different SVM kernels (e.g., 'rbf', 'poly') to capture more complex patterns.
   - Perform hyperparameter tuning using Grid Search to find the best values for parameters like `C` and `gamma`.

## 📦 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- google.colab (for running on Google Colab)

### Install Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/electric-vehicle-type-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd electric-vehicle-type-prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   ```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
