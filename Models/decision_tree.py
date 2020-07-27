import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

# Path to training data
train_path = '../Data/train.csv'

# Read csv into DataFrame
train_data = pd.read_csv(train_path)

# Create target object, y
y = train_data.SalePrice

# Choose features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify and fit model
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# Make predictions and calculate mean absolute error
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leafe_nodes: {:,.0f}".format(val_mae))

# Using best values for max_leaf_nodes. This is figure out with a loop to check
# different MAE values when different numbers of nodes are used
model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leafe_nodes: {:,.0f}".format(val_mae))

# Let's try using a random forest
model = RandomForestRegressor()
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(val_mae))

