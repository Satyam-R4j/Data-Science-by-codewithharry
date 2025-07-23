from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

housing = pd.read_csv("housing.csv")

housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                                labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    str_train_set = housing.iloc[train_index].drop('income_cat', axis =1)
    str_test_set = housing.iloc[test_index].drop('income_cat', axis =1)

housing = str_train_set.copy()

housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis = 1)

print(housing, housing_labels)


num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attirbs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("nums", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attirbs)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)


#Training the model

#linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print("The root mean square error for linear regression is ",lin_rmse)
lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(lin_rmse).describe())



#Decision Tree model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
dec_rmses = -cross_val_score(dec_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)


# print("The root mean square error for decision tree regression is ",dec_rmses)
print(pd.Series(dec_rmses).describe())




#Random forest  model
Random_for_reg = RandomForestRegressor()
Random_for_reg.fit(housing_prepared,housing_labels)
Random_for_preds = Random_for_reg.predict(housing_prepared)
Random_for_rmse = root_mean_squared_error(housing_labels, Random_for_preds)
# print("The root mean square error for Random Forest regression is ",Random_for_rmse)
Random_for_rmses = -cross_val_score(Random_for_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(Random_for_rmses).describe())






