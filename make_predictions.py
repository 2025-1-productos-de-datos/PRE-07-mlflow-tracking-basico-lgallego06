import mlflow
import pandas as pd
FILE_PATH = "data/winequality-red.csv"
df= pd.read_csv(FILE_PATH)
y=df['quality']
x=df.drop(columns=['quality'])

logged_model='runs:/140b611bb1404c799dda13601fbfbad5/model'
loaded_model = mlflow.pyfunc.log_model(logged_model)
y=loaded_model.predict(x)
print(y)
