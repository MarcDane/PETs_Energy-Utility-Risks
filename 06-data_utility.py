from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd

NR_OF_MEASUREMENTS = 10
    
def preprocess_dataset(dataset: str, is_benchmark: bool, epsilon: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    # created during 02-preprocessing.py
    train_dataset_csv = f'data/{dataset}/{dataset}_train.csv'
    validate_dataset_csv = f'data/{dataset}/{dataset}_val.csv'

    if is_benchmark: # benchmark working with raw dataset
        df_anonymized_data = pd.read_csv(train_dataset_csv)
        df_validate_data = pd.read_csv(validate_dataset_csv)
    else:
        # created in this function
        description_file = 'data_utility/temp_description.json'
        anonymized_dataset_csv = 'data_utility/temp_dataset.csv'

        # copied as used in 03-dataset_generation.py
        df = pd.read_csv(train_dataset_csv)
        threshold_value = df.nunique().max() + 1
        degree_of_bayesian_network = 2
        num_tuples_to_generate = len(df.index)
        categorical_attributes  = {column: True for column in df.columns.tolist()}
        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=train_dataset_csv, epsilon=epsilon, k=degree_of_bayesian_network, attribute_to_is_categorical=categorical_attributes)
        describer.save_dataset_description_to_file(description_file)
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
        generator.save_synthetic_data(anonymized_dataset_csv)

        df_anonymized_data = pd.read_csv(anonymized_dataset_csv)
        df_validate_data = pd.read_csv(validate_dataset_csv)

    
    results = []

    # create and return the data and labels (anonymized & validation)
    if dataset == 'adult':
        for df in [df_anonymized_data, df_validate_data]:

            # Convert to binary label
            df['income'] = df['income'].replace('<=50K', '0')
            df['income'] = df['income'].replace('>50K', '1')
            df['income'] = df['income'].astype(int)

            # Labels creation
            labels = df['income'].copy()
            df = df.drop(columns=['income'])

            # Set 'continuous' and 'categorical' features
            df[['age', 'capital_gain', 'capital_loss', 'hours_per_week']] = MinMaxScaler().fit_transform(df[['age', 'capital_gain', 'capital_loss', 'hours_per_week']])
            df = pd.get_dummies(df, columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])

            results.append(df)
            results.append(labels)
        
    if dataset == 'student':
        for df in [df_anonymized_data, df_validate_data]:

            # Set G3 to 0 or 1 (< 10 failed, >= 10 passed)
            df['G3'] = (df['G3'] >= 10).astype(int)

            # Labels creation
            labels = df['G3'].copy()
            df = df.drop(columns=['G3'])

            # Set 'continuous' and 'categorical' features
            continous_columns = ['age', 'absences', 'G1', 'G2']
            categorial_columns = [feature for feature in df.columns.to_list() if feature not in continous_columns]
            df[continous_columns] = MinMaxScaler().fit_transform(df[continous_columns])
            df = pd.get_dummies(df, columns = categorial_columns)

            results.append(df)
            results.append(labels)

    return tuple(results)

    
def logreg(df_anonymized_data: pd.DataFrame, df_anonymized_labels: pd.DataFrame, df_validate_data: pd.DataFrame, df_validate_labels: pd.DataFrame, dataset: str, dataset_type: str, epsilon: int):

    try:
        LogReg = LogisticRegression(max_iter=1000)
        LogReg.fit(df_anonymized_data, df_anonymized_labels.values.ravel())

        predictions = LogReg.predict(df_validate_data)
        report = classification_report(df_validate_labels.values.ravel(), predictions, output_dict=True)

        utility_csv = f'data_utility/results.csv'
        if os.path.exists(utility_csv) and os.path.getsize(utility_csv) > 0:
            header = False
        else:
            header = True

        result = {
            'dataset': dataset,
            'dataset_type': dataset_type,
            'epsilon': epsilon,
            'type': 'logreg',
            'accuracy': float(report.get('accuracy'))
        }
        pd.DataFrame([result]).to_csv(utility_csv, index=False, mode='a', header=header)
    except Exception as e:
        print(f'Error sampling LogReg: {str(e)}')

def knn(df_anonymized_data: pd.DataFrame, df_anonymized_labels: pd.DataFrame, df_validate_data: pd.DataFrame, df_validate_labels: pd.DataFrame, dataset: str, dataset_type: str, epsilon: int):
    try:
        NeighbourModel = KNeighborsClassifier(n_neighbors=8)
        NeighbourModel.fit(df_anonymized_data, df_anonymized_labels.values.ravel())
        report = classification_report(df_validate_labels, NeighbourModel.predict(df_validate_data), output_dict=True)
        utility_csv = f'data_utility/results.csv'
        if os.path.exists(utility_csv) and os.path.getsize(utility_csv) > 0:
            header = False
        else:
            header = True

        result = {
            'dataset': dataset,
            'dataset_type': dataset_type,
            'epsilon': epsilon,
            'type': 'knn',
            'accuracy': float(report.get('accuracy'))
        }
        pd.DataFrame([result]).to_csv(utility_csv, index=False, mode='a', header=header)
    except Exception as e:
        print(f'Error sampling knn: {str(e)}')


for index_measurement in range(NR_OF_MEASUREMENTS):

    for dataset in ['adult', 'student']:

        # Benchmark
        print(f'Benchmark Measurement {index_measurement} - {dataset}')
        df_anonymized_data, df_anonymized_labels, df_validate_data, df_validate_labels = preprocess_dataset(dataset=dataset, is_benchmark=True, epsilon=0)
        logreg(df_anonymized_data, df_anonymized_labels, df_validate_data, df_validate_labels, dataset, 'benchmark', 0)
        knn(df_anonymized_data, df_anonymized_labels, df_validate_data, df_validate_labels, dataset, 'benchmark', 0)

        epsilons = np.linspace(0, 20, num=101) # 0, 0.2, 0.4 ... 20
        for epsilon in epsilons:
            print(f'SYN+DP Measurement {index_measurement} - {dataset} - Epsilon {epsilon}')
            df_anonymized_data, df_anonymized_labels, df_validate_data, df_validate_labels = preprocess_dataset(dataset=dataset, is_benchmark=False, epsilon=epsilon)
            logreg(df_anonymized_data, df_anonymized_labels, df_validate_data, df_validate_labels, dataset, 'syn+dp', epsilon)
            knn(df_anonymized_data, df_anonymized_labels, df_validate_data, df_validate_labels, dataset, 'syn+dp', epsilon)

