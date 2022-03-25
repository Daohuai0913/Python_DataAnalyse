import pandas as pd

import pandas as pd

df = pd.DataFrame({'name': ['张蔷', '伍剑中', '贾文静', '宋文燕'], 'weight': [100, 200, 100, 100],
                   'gender': ['female', 'male', 'female', 'female']})
print(df)

print(df.rename(columns={'weight': '体重'}, copy=False))
print("----------------------")
print(df.rename(columns={'weight': '体重'}, copy=True))