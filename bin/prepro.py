import pandas as pd
import numpy as np

data = pd.read_csv("bank-additional.csv",
                   names=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                          'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                          'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'])
print(data)
