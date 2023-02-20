from sklearn.datasets import load_breast_cancer
import pandas as pd

def get_dataset():
    breast_dataset = load_breast_cancer()
    breast = pd.DataFrame(breast_dataset.data, columns=breast_dataset.feature_names)
    breast['y'] = breast_dataset.target
    test_dataset = breast.iloc[-100:,:]
    train_dataset = breast.iloc[:-100,:]
    train_dataset_dict = {}
    train_dataset_dict['A'] = train_dataset.iloc[:200,:]
    train_dataset_dict['B'] = train_dataset.iloc[200:,:]
    return test_dataset, train_dataset_dict