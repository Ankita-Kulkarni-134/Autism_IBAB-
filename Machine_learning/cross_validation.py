from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler


def cross_val_defined_lin(x_train, y_train):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # initialise the list to store the performance
    all_mse = []
    all_mae = []
    all_r2 = []
    all_accuracy = []

    # defining the fold
    for train_index, test_index in kf.split(x_train):
        x_train_val_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_val_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # scaling
        sk = StandardScaler()
        sk_x_train_val_fold = sk.fit_transform(x_train_val_fold)
        sk_x_val_fold = sk.transform(x_val_fold)

        # fit the model
        regressor = LinearRegression()
        regressor.fit(sk_x_train_val_fold, y_train_val_fold)

        # make the predictions
        y_pred_fold = regressor.predict(sk_x_val_fold)

        # check performance for linear
        mse = mean_squared_error(y_val_fold, y_pred_fold)
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        r2 = r2_score(y_val_fold, y_pred_fold)

        # check performance for logistic
        accuracy = accuracy_score(y_val_fold, y_pred_fold)

        # append the performance
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
        all_accuracy.append(accuracy)

    # printing the lists of the values are folds
    print('all_mse', all_mse)
    print('all_mae', all_mae)
    print('all_r2', all_r2)
    # logistic
    print('all_accuracy : ', all_accuracy)

    # take the average of performance
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    avg_r2 = np.mean(all_r2)
    print('avg_mse', avg_mse)
    print('avg_mae', avg_mae)
    print('avg_r2', avg_r2)

    # logistic
    avg_accuracy = np.mean(all_accuracy)
    print('avg_accuracy : ', avg_accuracy)
    return avg_mse, avg_mae, avg_r2, avg_accuracy, all_mse, all_mae, all_r2, all_accuracy


