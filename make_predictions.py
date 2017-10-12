# Load saved model
import pickle

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

# Read test file
import pandas as pd

test_df = pd.read_csv('data/test_potus_by_county.csv')

# Alter features to remain consistency in prediction, and create matrix

# Population growth is a multiplier on total population
test_df['Population growth'] = test_df['Population growth'] * test_df['Total population']

# Percentage of people with bachelors degree or higher can be quantified
test_df['% BachelorsDeg or higher'] = test_df['Total population'] * test_df['% BachelorsDeg or higher']

# Unemployment rate can be quanitified to number unemployed
test_df['Unemployment rate'] = test_df['Unemployment rate'] * test_df['Total population']

# Household growth is a multiplier on the number of households
test_df['House hold growth'] = test_df['House hold growth'] * test_df['Total households']

# Per capita income growth is a multiplier on per capita income
test_df['Per capita income growth'] = test_df['Per capita income growth'] * test_df['Per capita income']

X = test_df.as_matrix()

# Perform the prediction
import time

start = time.time()
y_pred = model.predict(X)
print('Time taken for prediction: %fs' % (time.time() - start))

# Save to predictions.csv
pred_df = pd.DataFrame(y_pred, columns=['Winner'])
pred_df.to_csv('predictions.csv', index=False)

# Run validations
from validate import run_basic_validations

run_basic_validations('predictions.csv')

