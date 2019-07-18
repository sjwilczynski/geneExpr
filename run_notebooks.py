import yaml
import os
import papermill as pm

with open('config.yaml', 'r') as f:
        config = yaml.load(f)

target_directory = 'notebook_results/' + config['target_directory']
os.makedirs(target_directory, exist_ok=True)

scoring = config['scoring']
cv = int(config['cv'])
cv_out = int(config['cv_out'])
cv_in = int(config['cv_in'])
n_iter = int(config['n_iter'])
n_jobs = int(config['n_jobs'])
notebooks = config['notebooks']

params_str = '_{}_{}_{}'.format(scoring, cv, n_iter)

for notebook in notebooks:
    os.chdir("notebooks/")
    pm.execute_notebook(
       notebook + '.ipynb',
       './../' + target_directory + notebook + params_str +'.ipynb',
       parameters = dict(scoring=scoring, cv=cv, cv_out=cv_out, cv_in=cv_in, n_iter=n_iter, n_jobs=n_jobs)
    )
    os.chdir("../")