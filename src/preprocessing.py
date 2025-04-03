import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(data_path):
    """ 
    Load CSV data.

    Parameters
    ----------
    data_path (str): Path to original data 

    Returns
    -------
    data (pd.DataFrmae): Dataframe of original data 
    """
    data = pd.read_csv(data_path)
    return data

def process_data(data, target_str, test_size, random_state, chi2percentile):
    """ 
    Preprocess data. Remove high VIF features, split, impute, scale, encode. 

    Parameters
    ----------
        - data (pd.DataFrame): Output from `load_data` function
        - target_str (str): Target variable name
        - test_size (float): Test size proportion
        - random_state (int): Random state
        - chi2percentile (float): 

    Returns 
    -------
    tuple of (train_new, test_new, clf)   
        - train_new
        - test_new
        - clf      
    
    """
    # Remove top 10 features with highest VIF
    X_df = data.drop(columns=target_str)
    X_numeric = X_df.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(len(X_numeric.columns))]
    vif_data = vif_data.sort_values(by='VIF', ascending=False)
    top_10_VIF = vif_data.head(10)["feature"].values

    X = data.drop(columns=list(top_10_VIF) + [target_str])
    y = data[target_str]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Combine X and y, impute missing, etc. 
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_y = train_data[target_str].values.reshape(-1, 1)
    test_y = test_data[target_str].values.reshape(-1, 1)

    impy = SimpleImputer(strategy="most_frequent")
    train_y = impy.fit_transform(train_y)
    test_y = impy.transform(test_y)

    train_data = train_data.drop(columns=[target_str])
    test_data = test_data.drop(columns=[target_str])

    # Pipelines
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=chi2percentile)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=["int", "float"])),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=["int", "float"])),
        ]
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit & transform
    clf.fit(train_data, train_y)
    train_new = clf.transform(train_data)
    test_new = clf.transform(test_data)

    # Convert to dfs
    train_new = pd.DataFrame(train_new)
    test_new = pd.DataFrame(test_new)
    train_new[target_str] = train_y.ravel()
    test_new[target_str] = test_y.ravel()


    return train_new, test_new, clf

def save_data(train_new, test_new, train_name, test_name, clf, clf_name):
    """ 
    Save processed data and pipeline.

    Parameters
    ----------
        - train_new 
        - test_new
        - test_strain_name
        - test_name
        - clf
        - clf_name

    Returns 
    -------
    Nothing     
    
    """
    train_new.to_csv(train_name, index=False)
    test_new.to_csv(test_name, index=False)

    with open(clf_name, "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["features"]
    data_path = params["data_path"]
    target_str = params["target_str"]
    test_size = params["test_size"]
    random_state = params["random_state"]
    chi2percentile = params["chi2percentile"]

    data = load_data(data_path)
    train_new, test_new, clf = process_data(
        data, target_str, test_size, random_state, chi2percentile
    )
    save_data(
        train_new,
        test_new,
        "data/best_cancer_processed_train_data.csv",
        "data/best_cancer_processed_test_data.csv",
        clf,
        "data/pipeline.pkl",
    )
