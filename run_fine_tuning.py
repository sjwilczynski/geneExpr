import yaml
import os
import papermill as pm

with open('config.yaml', 'r') as f:
        config = yaml.load(f)

target_directory = './notebook_results/' + config['target_directory']
os.makedirs(target_directory, exist_ok=True)

scoring = config['scoring']
cv = int(config['cv'])
n_iter = int(config['n_iter'])
n_jobs = int(config['n_jobs'])
notebook = config['notebook']

params_str = '{}_{}_{}'.format(scoring, cv, n_iter)

pm.execute_notebook(
   notebook + '.ipynb',
   target_directory + notebook + params_str +'.ipynb',
   parameters = dict(scoring=scoring, cv=cv, n_iter=n_iter, n_jobs=n_jobs)
)


