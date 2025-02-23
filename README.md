# Titanic Kaggle Competition Model

## Author: Alejandro Jiménez-González
### Date: 2025-02-23

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data](#data)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Description
This repository contains my current model for the Titanic Kaggle competition. It uses GridSearchCV to find the best parameters between `RandomForestClassifier` and `CatBoostClassifier`. The best model is then trained and used to predict the test set. The predictions are saved in a CSV file in the 'output' folder.

## Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/titanic-kaggle.git
    cd titanic-kaggle
    ```

2. **Set up the conda environment**:
    ```sh
    conda create --name titanic_env python=3.8
    conda activate titanic_env
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Data
The dataset used in this competition is provided by Kaggle and contains information about the passengers on the Titanic. The features include:

- `Pclass`: Passenger class
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation

### Download the Data
1. **Download the data from Kaggle**:
    - Go to the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) competition page on Kaggle.
    - Download the `train.csv` and `test.csv` files.

2. **Place the data files in the [data](http://_vscodecontentref_/0) directory**:
    - Create a [data](http://_vscodecontentref_/1) directory in the project root if it doesn't exist:
      ```sh
      mkdir data
      ```
    - Move the downloaded `train.csv` and `test.csv` files to the [data](http://_vscodecontentref_/2) directory.

## Model Training
The script performs the following steps:
1. **Data Preprocessing**: Handles missing values and encodes categorical variables.
2. **Feature Scaling**: Scales the features using `StandardScaler`.
3. **Grid Search**: Uses `GridSearchCV` to find the best parameters for `RandomForestClassifier` and `CatBoostClassifier`.
4. **Model Selection**: Compares the best scores and selects the best model.
5. **Model Training**: Trains the selected model with the best parameters.
6. **Prediction**: Uses the trained model to predict the test set.

## Usage
To run the script, use the following command:
```sh
python kaggle_titanic_model.py
```

## Results
The script outputs the best accuracy and parameters for both models. It also saves the predictions in a CSV file in the 'output' folder.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
