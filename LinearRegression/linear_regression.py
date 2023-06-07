import csv
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from fibonacci_method import fibonacci


data_dir = "dataset_solar_radiation.csv"


def read_data(data_dir):
   with open(data_dir, 'r') as f:
       reader = csv.DictReader(f)
       data = [row for row in reader]
   return data


def get_data(data):
   feature_names = [key for key in data[0].keys() if key not in [
       'Hourly_DateTime', 'Radiation', 'Log_Radiation']]
   features = [[float(row[key]) for key in feature_names] for row in data]
   radiation = [float(row['Radiation']) for row in data]
   return np.array(features), np.array(radiation)


def Loss(X, y, gradA, gradB):
    predictions = np.dot(gradA, X.T) + gradB
    errors = y - predictions
    squared_errors = np.square(errors)
    loss = np.mean(squared_errors)
    return loss


def dldw(X, y, W, b):
    predictions = np.dot(W, X.T) + b
    errors = y - predictions
    gradient = -2 * np.dot(X.T, errors) / X.shape[0]
    return gradient


def dldb(X, y, W, b):
    predictions = np.dot(W, X.T) + b
    errors = y - predictions
    gradient = -2 * np.sum(errors) / X.shape[0]
    return gradient


def gradient_descent(X, y, A, B):
   iteration = 0
   losses = []
   curr_loss = Loss(X, y, A, B)
   curr_dldw = dldw(X, y, A, B)
   curr_dldb = dldb(X, y, A, B)

   while np.abs(curr_loss - Loss(X, y, A - curr_dldw, B - curr_dldb)) > 1e-7:
       iteration += 1
       if iteration % 50 == 0:
           print(f"{iteration} | Loss: {curr_loss}")

       losses.append(curr_loss)

       beta = get_beta(X, y, A, B)
       A = A - beta * curr_dldw
       B = B - beta * curr_dldb
       curr_loss = Loss(X, y, A, B)
       curr_dldw = dldw(X, y, A, B)
       curr_dldb = dldb(X, y, A, B)
   return A, B, losses


def get_beta(X, y, W, b):

    def q(beta):
        return Loss(X, y, W - beta * dldw(X, y, W, b), b - beta * dldb(X, y, W, b))

    beta = fibonacci(q)
    return beta


def main():
   print("## Linear regression using gradient descent ##")
   X, y = get_data(read_data(data_dir))

   X = preprocessing.normalize(X)
   y = preprocessing.normalize([y])[0]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   A = np.random.randn(X.shape[1])
   b = np.random.randn()

   A, b, losses = gradient_descent(X_train, y_train, A, b)
   print("Loss on test set:", Loss(X_test, y_test, A, b))
   print(f'\nA: {A}')
   print(f'b: {b}')
   check(X, y)


def check(X, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    error = mean_squared_error(y_test, y_pred)
    print('\n## Check with sklearn ##')
    print("Mean Squared Error:", error)


if __name__ == "__main__":
   main()
