import pandas as pd
from sqlite3 import connect

conn = connect('/tmp/weightsRate/weightsRate_storage.db')

def batch_params():
    params = dict()
    params["excWeight"] = [0, 5e-3]
    params["inhWeightScale"] = [0.1, 10]
    params["gnabar"] = [1e-4, 1e-2]
    params["gkbar"] = [1e-4, 1e-2]
    params["ukcc2"] = [1e-6, 1]
    params["unkcc1"] = [1e-6, 1]
    params["pmax"] = [1e-6, 100]
    params["gpas"] = [0, 0.0001]
    return params


params = batch_params()

query = """SELECT 
    trial_values.trial_id,
    trial_values.objective,
    trial_values.value AS trial_value"""

for k in params:
    query += f",\nMAX(CASE WHEN trial_params.param_name = '{k}' THEN trial_params.param_value END) AS {k}"

query += """\nFROM 
    trial_values
JOIN 
    trial_params ON trial_values.trial_id = trial_params.trial_id
GROUP BY 
    trial_values.trial_id, trial_values.objective, trial_values.value;
"""


df = pd.read_sql(query, conn)
