import json
import pandas as pd

with open('results/test3/sweep_results.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
erm_acc = df.groupby(['scenario', 'seed'])['acc'].mean()['erm'].to_dict()
df['erm_acc'] = df['seed'].map(erm_acc)
df['improvement'] = df['acc'] - df['erm_acc']

bound_df = df[(df['scenario'] == 'proposed') & (df['objective'] == 'bound')]
heuristic_df = df[(df['scenario'] == 'proposed') & (df['objective'] == 'heuristic')]

bound_res = bound_df.groupby(['reg_coeff', 'reg_alpha', 'reg_beta'])['improvement'].mean().reset_index().sort_values('improvement', ascending=False)
heur_res = heuristic_df.groupby(['reg_coeff', 'reg_alpha', 'reg_beta'])['improvement'].mean().reset_index().sort_values('improvement', ascending=False)

print("--- BOUND OBJECTIVE IMPROVEMENTS ---")
print(bound_res.head(10))
print("\n--- HEURISTIC OBJECTIVE IMPROVEMENTS ---")
print(heur_res.head(10))

