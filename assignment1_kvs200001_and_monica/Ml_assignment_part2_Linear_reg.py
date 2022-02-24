import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class Linear_Regression:
  def __init__(self, path):
    self.path = path
    self.data = pd.read_csv(path)
    self.columns = ['No','Cement','Slag','Fly_Ash','Water','SP','Coarse_Agger','Fine_Aggr','Slump','Flow','MPA']
    self.data.columns = self.columns
    self.preprocessing()

  def preprocessing(self):
    self.data.dropna(axis = 0, how = 'any', thresh = None, inplace = True)
    self.data.drop_duplicates(inplace=True)
    correlation_matrix=self.data.corr()
    sns.heatmap(data=correlation_matrix, annot=True)
    #print(correlation_matrix)
    s = StandardScaler()
    self.preprocessed_data = pd.DataFrame(s.fit(self.data).fit_transform(self.data), columns=self.data.columns)

  def run_linear_regression(self):
    X = self.preprocessed_data[['Cement','Slag','Fly_Ash','Water','SP','Coarse_Agger','Fine_Aggr']]
    Y = self.preprocessed_data['Slump']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=5)
    model = SGDRegressor()
    model.fit(X_train, Y_train)
    self.Y_predicted = model.predict(X_test)
    self.calculate_error_statistics(Y_test, self.Y_predicted)

  def calculate_error_statistics(self, Y, Y_predicted):
    self.mse = mean_squared_error(Y, Y_predicted)
    self.rmse = (np.sqrt(self.mse))
    self.r2 = r2_score(Y, Y_predicted)

  def print_statistics(self):
    print('RMSE is {}'.format(self.rmse))
    print('R2 score is {}'.format(self.r2))
    print('MSE is {}'.format(self.mse))


if __name__ == "__main__":
  model = Linear_Regression('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data')
  model.run_linear_regression()
  model.print_statistics()