import random as rand
import pandas as pd
import math

data_size_test = 1000
data_size = 100000

coord = []
coord_test = []

for i in range(1, data_size):
    r = rand.uniform(-20, 20)
    coord.append((r, math.sin(r)))

for i in range(1, data_size_test):
    r = rand.uniform(-20, 20)
    coord_test.append((r, math.sin(r)))

df = pd.DataFrame(coord, columns=['X', 'Y'])
df_test = pd.DataFrame(coord_test, columns=['X', 'Y'])

df.to_csv(r"FunctionData.csv", index=False)
df_test.to_csv(r"FunctionTestData.csv", index=False)