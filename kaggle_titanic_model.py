# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Load the dataset
train_dataset = pd.read_csv(f'{data_folder}/train.csv')
test_dataset = pd.read_csv(f'{data_folder}/test.csv')

# Selecting relevant features and dependent variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_dataset.loc[:, features].values  # All columns specified in features
y = train_dataset.loc[:, 'Survived'].values  # The 'Survived' column

# Handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X[:, 2:3] = imputer.fit_transform(X[:, 2:3])  # Age column
X_test = test_dataset.loc[:, features].values
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])  # Age column in test set

imputer = SimpleImputer(strategy='most_frequent')
X[:, 6:7] = imputer.fit_transform(X[:, 6:7])  # Embarked column
X_test[:, 6:7] = imputer.transform(X_test[:, 6:7])  # Embarked column in test set

# Encoding categorical variables
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])  # Sex column
X_test[:, 1] = labelencoder_X.transform(X_test[:, 1])  # Sex column in test set

labelencoder_X_embarked = LabelEncoder()
X[:, 6] = labelencoder_X_embarked.fit_transform(X[:, 6])  # Embarked column
X_test[:, 6] = labelencoder_X_embarked.transform(X_test[:, 6])  # Embarked column in test set

# Applying OneHotEncoder to both 'Sex' and 'Embarked' columns
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [1, 6])
    ],
    remainder='passthrough'
)
X = ct.fit_transform(X)
X_test = ct.transform(X_test)

# Avoiding the dummy variable trap
X = X[:, 1:]
X_test = X_test[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# Parameters for Grid Search
rf_parameters = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

catboost_parameters = {
    'iterations': [50, 100, 200],
    'depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7]
}

# Grid Search for RandomForestClassifier
print("Starting Grid Search for RandomForestClassifier")
rf_classifier = RandomForestClassifier(random_state=0)
rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_parameters, scoring='accuracy', cv=10, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
rf_best_accuracy = rf_grid_search.best_score_
rf_best_parameters = rf_grid_search.best_params_
print("RandomForest Best Accuracy: {:.2f} %".format(rf_best_accuracy*100))
print("RandomForest Best Parameters:", rf_best_parameters)

# Grid Search for CatBoostClassifier
print("Starting Grid Search for CatBoostClassifier")
catboost_classifier = CatBoostClassifier(random_state=0, verbose=0)
catboost_grid_search = GridSearchCV(estimator=catboost_classifier, param_grid=catboost_parameters, scoring='accuracy', cv=10, n_jobs=-1)
catboost_grid_search.fit(X_train, y_train)
catboost_best_accuracy = catboost_grid_search.best_score_
catboost_best_parameters = catboost_grid_search.best_params_
print("CatBoostClassifier Best Accuracy: {:.2f} %".format(catboost_best_accuracy*100))
print("CatBoostClassifier Best Parameters:", catboost_best_parameters)

# Compare and select the best model
if catboost_best_accuracy > rf_best_accuracy:
    best_classifier = CatBoostClassifier(**catboost_best_parameters, random_state=0, verbose=0)
    print("Using CatBoostClassifier")
else:
    best_classifier = RandomForestClassifier(**rf_best_parameters, random_state=0)
    print("Using RandomForestClassifier")

# Train the best model with the best parameters
best_classifier.fit(X_train, y_train)

# Making the Confusion Matrix with the best model
y_pred = best_classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
print(cm)

# Predicting the Test set results with the best model
y_pred_test = best_classifier.predict(X_test)

# Creating the submission file
submission = pd.DataFrame({'PassengerId': test_dataset['PassengerId'], 'Survived': y_pred_test})
submission.to_csv(f'{data_folder}/output/submission.csv', index=False)