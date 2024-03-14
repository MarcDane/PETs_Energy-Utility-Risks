import pandas as pd

def adult_preprocessing():
    column_names = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'native_country',
        'income'
    ]

    # region Training Set
    df = pd.read_csv('data/adult/adult.data', sep=',\s', names=column_names, na_values=['?'], engine='python')

    df = df.drop(columns=['fnlwgt', 'education_num']) # superfluous/duplicate columns
    df = df.dropna()

    df.to_csv('data/adult/adult_train.csv', index=False)
    # endregion

    # region Validation Set
    df = pd.read_csv('data/adult/adult.test', sep=',\s', names=column_names, na_values=['?'], engine='python')

    df = df.drop(columns=['fnlwgt', 'education_num']) # superfluous/duplicate columns
    df = df.dropna()
    df['income'] = df['income'].replace('<=50K.', '<=50K') # match values with training set
    df['income'] = df['income'].replace('>50K.', '>50K') # match values with training set

    df.to_csv('data/adult/adult_val.csv', index=False)
    # endregion

def student_preprocessing():
    df = pd.read_csv('data/student/student-por.csv', sep=';')

    df_train = df.sample(frac=0.8) # 80% of instances
    df_control = df.drop(df_train.index) # assign remaning rows to control

    # region Training Set
    df_train.to_csv('data/student/student_train.csv', index=False)
    # endregion

    # region Validation Set
    df_control.to_csv('data/student/student_val.csv', index=False)
    # endregion
    
print(f'Cleaning Adult ...')
adult_preprocessing()

print(f'Cleaning Student ...')
student_preprocessing()
