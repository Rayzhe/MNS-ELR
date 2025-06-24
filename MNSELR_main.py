import os

import numpy as np
import scipy

from Preprocessing import process_eeg_covariance
from MNSELR_model import Class_L21L1

# Optional for Bayesian optimization:
# from skopt import BayesSearchCV
# from skopt.space import Real
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score

def main():
    # ======== 1. Load your data here ==========
    filepath = f'D:/HH/data/data_1/'
    files = os.listdir(filepath)
    file_names = [file_name.split('.')[0] for file_name in files]
    n_numbers = len(files)
    cnt = np.zeros([n_numbers])
    cnt_test = np.zeros([n_numbers])
    accTest = np.zeros([n_numbers])
    lambda12 = np.zeros([n_numbers])
    acc_train = np.zeros([n_numbers])
    results = []
    for k in range(n_numbers):
        f = scipy.io.loadmat(filepath + files[k])
        subject = file_names[k]
        dataTrain = f['X_train']
        dataTest = f['X_test']
        labelTrain = f['Y_train']
        labelTest = f['Y_test']
        y_train = labelTrain.squeeze()
        y_test = labelTest.squeeze()

        # ======== 2. Preprocessing: Covariance, Whitening, Log mapping ==========
        RsTrain, RsTest = process_eeg_covariance(dataTrain, dataTest)

        # ======== 3. Train L21L1 model ==========
        clf = Class_L21L1(lambda1=1.55, lambda21=3.05)
        clf.fit(RsTrain, y_train)

        # ======== 4. Predict and evaluate ==========
        acc = clf.score(RsTest, y_test)
        print(f"Test accuracy: {acc:.4f}")
        results.append(acc)
    print(np.mean(results))

    # ======== 5. (Optional) Use Bayesian Optimization for hyperparameter tuning ========
    """
    # Uncomment to use Bayesian Optimization for lambda1 and lambda21
    from skopt import BayesSearchCV
    from skopt.space import Real
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    model = Class_L21L1(lambda1=1, lambda21=1)
    search_spaces = {
        'lambda1': Real(0.01, 10, prior='uniform'),
        'lambda21': Real(0.01, 10, prior='uniform')
    }
    n_splits = 10  # keep balance of labels
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)
    opt = BayesSearchCV(
        model, search_spaces, n_iter=40, scoring="accuracy", cv=cv, random_state=99, n_jobs=-1
    )
    opt.fit(RsTrain, y_train)

    y_pred = opt.best_estimator_.predict(RsTest)
    accuracy = accuracy_score(y_test, y_pred)
    best_params = opt.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Bayesian Optimization Test accuracy: {accuracy:.4f}")
    """

if __name__ == '__main__':
    main()
