from anonymeter.evaluators import InferenceEvaluator
from anonymeter.evaluators import LinkabilityEvaluator
from anonymeter.evaluators import SinglingOutEvaluator
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from typing import Tuple
import numpy as np
import os
import pandas as pd

NR_OF_MEASUREMENTS = 10

# Get the Training, Anonymized and Control Datasets
def get_datasets(dataset: str, epsilon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # created during 02-preprocessing.py
    train_dataset_csv = f'data/{dataset}/{dataset}_train.csv'
    control_dataset_csv = f'data/{dataset}/{dataset}_val.csv'

    # created in this function
    description_file = 'anonymization_effectiveness/temp_description.json'
    anonymized_dataset_csv = 'anonymization_effectiveness/temp_dataset.csv'

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

    df_train = pd.read_csv(train_dataset_csv)
    df_anonymized = pd.read_csv(anonymized_dataset_csv)
    df_control = pd.read_csv(control_dataset_csv)
    return (df_train, df_anonymized, df_control)

def measure_risks(df_train: pd.DataFrame, df_anonymized: pd.DataFrame, df_control: pd.DataFrame, dataset: str, epsilon: int):
    results = []

    # region Single Out
    print(f'Single Out Measurement - {dataset} - Epsilon {epsilon}')
    evaluator = SinglingOutEvaluator(ori=df_train, syn=df_anonymized, control=df_control, n_attacks=3000)
    try:
        evaluator.evaluate(mode='univariate')
        res = evaluator.results()
        result = {
            'dataset': dataset,
            'attack': 'singling_out',
            'epsilon': epsilon,
            'secret_column': None,
            'success_rate_main': res.attack_rate.value,
            'success_rate_baseline': res.baseline_rate.value,
            'success_rate_control': res.control_rate.value,
            'risk': res.risk().value
        }
        results.append(result)
    except RuntimeError as ex: 
        print(f'Singling out evaluation failed with {ex}. Please re-run this cell. For more stable results increase `n_attacks`')
    # endregion          
    
    # region Linkability
    print(f'Linkability Measurement - {dataset} - Epsilon {epsilon}')
    auxiliary_columns = [] # Columns to assume are well-known by an attacker
    if dataset == 'adult':
        auxiliary_columns = ['workclass', 'race', 'sex', 'age']
        nr_attacks = 3000
    if dataset == 'student':
        auxiliary_columns = ['sex', 'age', 'school', 'guardian']
        nr_attacks = 100 # for a smaller dataset
    evaluator = LinkabilityEvaluator(ori=df_train, syn=df_anonymized, control=df_control, n_attacks=nr_attacks, aux_cols=auxiliary_columns, n_neighbors=8)
    evaluator.evaluate(n_jobs=-2) # -2 = all cores execept one
    res = evaluator.results()
    result = {
            'dataset': dataset,
            'attack': 'linkability',
            'epsilon': epsilon,
            'secret_column': None,
            'success_rate_main': res.attack_rate.value,
            'success_rate_baseline': res.baseline_rate.value,
            'success_rate_control': res.control_rate.value,
            'risk': res.risk().value
        }
    results.append(result)
    # endregion

    # region Inference
    print(f'Inference Measurement - {dataset} - Epsilon {epsilon}')
    columns = df_train.columns
    for secret in columns:
        aux_cols = [col for col in columns if col != secret]            
        evaluator = InferenceEvaluator(ori=df_train, syn=df_anonymized, control=df_control, aux_cols=aux_cols, secret=secret, n_attacks=nr_attacks)
        evaluator.evaluate(n_jobs=-2) # -2 = all cores execept one
        res = evaluator.results()
        result = {
            'dataset': dataset,
            'attack': 'inference',
            'epsilon': epsilon,
            'secret_column': secret,
            'success_rate_main': res.attack_rate.value,
            'success_rate_baseline': res.baseline_rate.value,
            'success_rate_control': res.control_rate.value,
            'risk': res.risk().value
        }
        results.append(result)
    # endregion
            
    risk_csv = f'anonymization_effectiveness/risks_{dataset}.csv'
    if os.path.exists(risk_csv) and os.path.getsize(risk_csv) > 0:
        header = False
    else:
        header = True

    pd.DataFrame(results).to_csv(risk_csv, index=False, mode='a', header=header)


        
epsilons = np.linspace(0, 20, num=101) # 0, 0.2, 0.4 ... 20
for epsilon in epsilons:
    for dataset in ['adult', 'student']:
        df_train, df_anonymized, df_control = get_datasets(dataset=dataset, epsilon=epsilon)
        for _ in range(NR_OF_MEASUREMENTS):
            measure_risks(
                df_train = df_train,
                df_anonymized = df_anonymized,
                df_control = df_control,
                dataset = dataset,
                epsilon = epsilon
            )