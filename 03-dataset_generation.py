from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
import pandas as pd
import pyRAPL
import time

NR_OF_MEASUREMENTS = 10

def create_dataset_base(input_data_file: str, output_description_file: str, output_data_file: str, pyrapl_label: str, pyrapl_output_file: str, epsilon: int):
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput(pyrapl_output_file)

    df = pd.read_csv(input_data_file)

    threshold_value = df.nunique().max() + 1 # Consider all attributes as 'Categorical'
    degree_of_bayesian_network = 2
    num_tuples_to_generate = len(df.index)
    categorical_attributes  = {column: True for column in df.columns.tolist()} # Consider all attributes as 'Categorical'

    df = None # clear loaded memory
    
    with pyRAPL.Measurement(label=f'{pyrapl_label}__describe', output=csv_output):
        describer = DataDescriber(category_threshold=threshold_value)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data_file, epsilon=epsilon, k=degree_of_bayesian_network, attribute_to_is_categorical=categorical_attributes)
        describer.save_dataset_description_to_file(output_description_file)
    csv_output.save()
    
    with pyRAPL.Measurement(label=f'{pyrapl_label}__generate', output=csv_output):
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, output_description_file)
        generator.save_synthetic_data(output_data_file)
    csv_output.save()


for index_measurement in range(NR_OF_MEASUREMENTS):
    for dataset in ['adult', 'student']:
        for epsilon in [0.0, 0.1, 20]: # SYN == 0.0,  DP == 0.1 & 20
            print(f'Measurement {index_measurement} -- {dataset} -- Epsilon {epsilon} ...')

            create_dataset_base(
                input_data_file = f'data/{dataset}/{dataset}_train.csv',
                output_description_file = f'dataset_generation/description_{dataset}.json',
                output_data_file = f'dataset_generation/data_{dataset}.csv',
                pyrapl_label = dataset,
                pyrapl_output_file = 'energy/dataset_generation.csv',
                epsilon = epsilon
            )

            time.sleep(10)