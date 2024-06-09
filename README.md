# Rock vs Mine Prediction System

## Overview

The **Rock vs Mine Prediction System** is a machine learning project designed to classify objects as either rocks or mines based on sonar data. This project uses Logistic Regression for predictive modeling and demonstrates the application of data preprocessing, model training, and evaluation techniques.

## Features

- **Data Preprocessing**: Efficient handling and preparation of sonar data for training the model.
- **Model Training**: Utilizes Logistic Regression for training the model on the preprocessed data.
- **Model Evaluation**: Evaluates the model's performance using accuracy metrics on both training and test datasets.
- **Prediction System**: Allows users to input new data to predict whether the object is a rock or a mine.

## Technologies Used

- **Python**: Programming language used for developing the model and predictive system.
- **Pandas**: Library used for data manipulation and analysis.
- **NumPy**: Library used for numerical computations.
- **scikit-learn**: Machine learning library used for model training and evaluation.

## Project Setup

### Prerequisites

- **Python 3.x**: Ensure you have Python installed. [Download Python](https://www.python.org/downloads/)
- **Pandas**: Install using `pip install pandas`
- **NumPy**: Install using `pip install numpy`
- **scikit-learn**: Install using `pip install scikit-learn`

### Installation and Running the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SahilInSyntax/rock-vs-mine-prediction.git
   cd rock-vs-mine-prediction
   ```

2. **Install Dependencies**:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. **Download the Dataset**:
   Ensure you have the sonar data file named `sonar_data.csv` in the project directory.

4. **Run the Python Script**:
   
   ```bash
   python rock_vs_mine_prediction.py
   ```

### Usage

- **Training the Model**:
  The script will automatically split the data into training and test sets, train the Logistic Regression model, and print the accuracy on both training and test datasets.

- **Making Predictions**:
  The script includes an example of making a prediction for a single data instance. You can modify the `input_data` variable to predict for other instances.

### Example Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
sonar_data = pd.read_csv('sonar_data.csv', header=None)

# Separate features and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)

print(f'Accuracy on training data: {train_accuracy}')
print(f'Accuracy on test data: {test_accuracy}')

# Predict on new data
input_data = (0.0408, 0.0653, 0.0397, 0.0604, 0.0496, 0.1817, 0.1178, 0.1024, 0.0583, 0.2176, 
              0.2459, 0.3332, 0.3087, 0.2613, 0.3232, 0.3731, 0.4203, 0.5364, 0.7062, 0.8196, 
              0.8835, 0.8299, 0.7609, 0.7605, 0.8367, 0.8905, 0.7652, 0.5897, 0.3037, 0.0823, 
              0.2787, 0.7241, 0.8032, 0.8050, 0.7676, 0.7468, 0.6253, 0.1730, 0.2916, 0.5003, 
              0.5220, 0.4824, 0.4004, 0.3877, 0.1651, 0.0442, 0.0663, 0.0418, 0.0475, 0.0235, 
              0.0066, 0.0062, 0.0129, 0.0184, 0.0069, 0.0198, 0.0199, 0.0102, 0.0070, 0.0055)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_as_numpy_array)

print(f'The object is a {"Rock" if prediction[0] == "R" else "Mine"}')
```

### Future Enhancements

- **Advanced Models**: Implement more complex models such as Random Forest or Neural Networks for potentially better accuracy.
- **Feature Engineering**: Apply feature engineering techniques to improve model performance.
- **Web Interface**: Develop a web interface to allow users to input data and get predictions without running the script.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions regarding this project, please contact [sahilmanjrekar2003@gmail.com].

---