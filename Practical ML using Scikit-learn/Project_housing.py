from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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




