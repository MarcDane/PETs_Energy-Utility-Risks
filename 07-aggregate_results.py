import matplotlib.pyplot as plt
import pandas as pd

# pd.set_option('display.max_rows', None) # to display all rows within df's

# Energy Idle
df = pd.read_csv('energy/idle.csv')
mean_duration = df['duration'].mean()
mean_pkg = df['pkg'].mean()

mean_duration *= (10**-6) # seconds
mean_pkg *= (10**-6) # joule

idle_energy = mean_pkg / mean_duration # Joule / sec
print(f'Idle Energy Usage: {idle_energy} (Joule / sec)\n')


# Energy DP-Query - Table
df = pd.read_csv('energy/dp.csv')
df = df.groupby(['label'])[['duration', 'pkg']].mean().reset_index()
df[['duration', 'pkg']] = df[['duration', 'pkg']] / 1000000
df['energy_usage'] = df['pkg'] - (idle_energy * df['duration'])
print(f'Energy Usage: DP (Query-Response)\n{df}\n')


# Energy SYN Dataset + DP Dataset - Table
df = pd.read_csv('energy/dataset_generation.csv')
df = df.groupby(['label'])[['duration', 'pkg']].mean().reset_index()
df[['duration', 'pkg']] = df[['duration', 'pkg']] / 1000000
df['energy_usage'] = df['pkg'] - (idle_energy * df['duration'])

df[['prefix', 'task']] = df['label'].str.split('__', expand=True)
df = df.pivot(index='prefix', columns='task', values=['duration', 'energy_usage'])
df.columns = [f'{col[1]}_{col[0]}' for col in df.columns.values]
df = df.reset_index()
df['total_duration'] = df['describe_duration'] + df['generate_duration']
df['total_energy_usage'] = df['describe_energy_usage'] + df['generate_energy_usage']
print(f'Energy Usage: SYN & DP Dataset\n{df}\n')


# Data Utility
df = pd.read_csv('data_utility/results.csv')
df = df.groupby(['dataset', 'dataset_type', 'epsilon', 'type'])['accuracy'].mean().reset_index()
df_benchmark = df[df['dataset_type'] == 'benchmark'].reset_index()
df_syn = df[(df['dataset_type'] == 'syn+dp') & (df['epsilon'] == 0.0)].reset_index()
df_dp = df[(df['dataset_type'] == 'syn+dp') & (df['epsilon'] != 0.0)].reset_index()

print(f'Data Utility: Benchmark Test\n{df_benchmark}\n')
print(f'Data Utility: SYN Test\n{df_syn}\n')
print(f'Data Utility: DP Test\n{df_dp}\n')

pivot_df = df.pivot_table(index='epsilon', columns=['dataset', 'type'], values='accuracy')
fig, ax = plt.subplots(figsize=(10, 6))
mapping_name = {
    'adult': '(DP) Census',
    'student': '(DP) Student'
}
mapping_type = {
    'knn': 'k-NN',
    'logreg': 'LogReg'
}
colors = ['blue', 'green', 'red', 'purple']
color_index = 0
lines = ['-', '--', '-.', ':']
for (dataset, model_type), values in pivot_df.items():
    dataset_name = mapping_name.get(dataset, dataset)
    type_name = mapping_type.get(model_type, model_type)
    ax.plot(pivot_df.index, values, label=f'{dataset_name} - {type_name}', linestyle=lines[color_index], color=colors[color_index])
    color_index = (color_index + 1) % len(colors)

ax.set_xlabel('Epsilon', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
ax.legend(fontsize='large')
plt.grid(True)
plt.savefig('results_output/data-utility__dp.png')
# plt.show()



'''
For Anonymization Effectiveness, if the rate of the MAIN attack is lower than the BASELINE attack,
the risk/analysis cannot be trusted, as the MAIN attack is weaker than just guessing/naive attack.
For this, the attacks/measurements that meet this criteria are ignored/invalidated
'''

# Anonymization Effectiveness - Adults
df = pd.read_csv('anonymization_effectiveness/risks_adult.csv')
df = df.fillna(0)
df = df.groupby(['dataset', 'attack', 'epsilon', 'secret_column'])['risk'].mean().reset_index()

df_linkability = df[(df['attack'] == 'linkability')].reset_index(drop=True)
print(f'Anonymization Effectiveness - Adults (Linkability): \n{df_linkability}\n')

df_singling_out = df[(df['attack'] == 'singling_out')].reset_index(drop=True)
print(f'Anonymization Effectiveness - Adults (Singling Out): \n{df_singling_out}\n')

df_inference = df[(df['attack'] == 'inference')].reset_index(drop=True)
# print(f'Anonymization Effectiveness - Adults (Inference): \n{df_inference}\n')
df_inference = df_inference.groupby(['dataset', 'attack', 'epsilon'])['risk'].mean().reset_index()
print(f'Anonymization Effectiveness - Adults (Inference Risk): \n{df_inference}\n')
df_adults_inference = df_inference.drop(0)

# Anonymization Effectiveness - Students
df = pd.read_csv('anonymization_effectiveness/risks_student.csv')
df = df.fillna(0)
df = df.groupby(['dataset', 'attack', 'epsilon', 'secret_column'])['risk'].mean().reset_index()

df_linkability = df[(df['attack'] == 'linkability')].reset_index(drop=True)
print(f'Anonymization Effectiveness - Students (Linkability): \n{df_linkability}\n')

df_singling_out = df[(df['attack'] == 'singling_out')].reset_index(drop=True)
print(f'Anonymization Effectiveness - Students (Singling Out): \n{df_singling_out}\n')

df_inference = df[(df['attack'] == 'inference')].reset_index(drop=True)
# print(f'Anonymization Effectiveness - Students (Inference): \n{df_inference}\n')
df_inference = df_inference.groupby(['dataset', 'attack', 'epsilon'])['risk'].mean().reset_index()
print(f'Anonymization Effectiveness - Students (Inference Risk): \n{df_inference}\n')
df_students_inference = df_inference.drop(0)


plt.figure(figsize=(10, 6))
plt.scatter(df_students_inference['epsilon'], df_students_inference['risk'], color='blue', label='DP - Students', alpha=0.7)
plt.scatter(df_adults_inference['epsilon'], df_adults_inference['risk'], color='red', label='DP - Census', alpha=0.7)

# plt.title('Inference Risk')
plt.xlabel('Epsilon', fontsize=18)
plt.ylabel('Inference Risk', fontsize=18)
plt.grid(True)
plt.legend(fontsize='large')
plt.savefig('results_output/inference_risk__dp.png')
