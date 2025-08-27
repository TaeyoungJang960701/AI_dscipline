import joblib
import pandas as pd

ourmodel = joblib.load('python_analysis/analysis04/mymodel.model')
new_df = pd.DataFrame({'Income':[44,44,44], 'Advertising':[6,3,11], 'Price':[105,88,77], 'Age':[33,55,22]})
new_pred = ourmodel.predict(new_df)
print(f'Sales 예측 결과:\n{new_pred}')
# 0     8.761168
# 1     8.303319
# 2    11.501090

# web으로 만들면 많은 사람이 쓸 수 있다. income... data를 input 받아서 예측 결과값 반환.