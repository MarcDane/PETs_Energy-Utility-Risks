from collections import Counter
import numpy as np
import opendp.prelude as dp
import pandas as pd
import pyRAPL
import time
dp.enable_features('contrib')

NR_OF_MEASUREMENTS = 10

def dp_laplace_mean(input_data_file: str, column: str, pyrapl_label: str, pyrapl_output_file: str):
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput(pyrapl_output_file)

    with pyRAPL.Measurement(label=pyrapl_label, output=csv_output):

        df = pd.read_csv(input_data_file)
        column_names = df.columns.to_list()
        bounds = (df[column].min(), df[column].max())
        float_bounds = tuple(map(float, bounds))
        df = None

        with open(input_data_file) as input_file:
            next(input_file) # skip csv headers row
            data = input_file.read()

        preprocessor = (
            dp.t.make_split_dataframe(separator=',', col_names=column_names) >>
            dp.t.make_select_column(key=column, TOA=str) >>
            dp.t.then_cast_default(TOA=float)
        )

        laplace_count = preprocessor >> dp.t.then_count() >> dp.m.then_base_discrete_laplace(scale=1.)
        data_count = laplace_count(data)

        laplace_mean = (
            preprocessor >>
            dp.t.then_clamp(bounds=float_bounds) >>
            dp.t.then_resize(size=data_count, constant=20.) >>
            dp.t.then_mean() >>
            dp.m.then_laplace(scale=1.)
        )

        mean = laplace_mean(data)
        # print(mean)

    csv_output.save()



def dp_randomized_response_mean(input_data_file: str, column: str, probability: float, pyrapl_label: str, pyrapl_output_file: str):
    pyRAPL.setup()
    csv_output = pyRAPL.outputs.CSVOutput(pyrapl_output_file)

    with pyRAPL.Measurement(label=pyrapl_label, output=csv_output):

        df = pd.read_csv(input_data_file)
        size = len(df.index)
        categories = sorted(df[column].unique())
        actual_distribution = (df[column].value_counts(normalize=True).sort_index()).to_list()
        df = None

        # Generate responses with noise based on probability
        rr_mean = dp.m.make_randomized_response(categories=categories, prob=probability)
        responses = []
        for _ in range(size):
            response = np.random.choice(a=categories, p=actual_distribution)
            responses.append(rr_mean(response))    
        counter = Counter(responses)
        dp_distribution = [counter[category] / size for category in categories]

        # 'De-bias' responses
        dp_distribution = np.array(dp_distribution)
        assert all(dp_distribution >= 0) and abs(sum(dp_distribution) - 1) < 1e-6
        assert 0 <= probability <= 1
        k = len(dp_distribution)
        debiased_dp_distribution = (dp_distribution * (k - 1) + probability - 1) / (probability * k - 1)

        # print(list(dp_distribution.round(5)))
        # print(list(debiased_dp_distribution.round(5)))

    csv_output.save()



for index_measurement in range(NR_OF_MEASUREMENTS):

    for dataset in ['adult', 'student']:
        print(f'Laplace Measurement {index_measurement} - {dataset} ...')
        dp_laplace_mean(
            input_data_file = f'data/{dataset}/{dataset}_train.csv',
            column = 'age',
            pyrapl_label = f'laplace_{dataset}',
            pyrapl_output_file = 'energy/dp.csv'
        )
        time.sleep(2)

        print(f'RR Measurement {index_measurement} - {dataset} ...')
        dp_randomized_response_mean(
            input_data_file = f'data/{dataset}/{dataset}_train.csv',
            column = 'age',
            probability = 0.6,
            pyrapl_label = f'rr_{dataset}',
            pyrapl_output_file = 'energy/dp.csv'
        )
        time.sleep(2)
