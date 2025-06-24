import numpy as np
from Preprocessing import process_eeg_covariance
from MNSELR_model import Class_L21L1

# Optional for Bayesian optimization:
# from skopt import BayesSearchCV
# from skopt.space import Real
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score

def main():
    # ======== 1. Load your data here ==========
    n_trials_train, n_trials_test = 50, 20
    n_samples, n_channels = 750, 60
    dataTrain = np.random.randn(n_trials_train, n_samples, n_channels)
    dataTest = np.random.randn(n_trials_test, n_samples, n_channels)
    y_train = np.random.choice([-1, 1], size=n_trials_train)
    y_test = np.random.choice([-1, 1], size=n_trials_test)

    # ======== 2. Preprocessing: Covariance, Whitening, Log mapping ==========
    RsTrain, RsTest = process_eeg_covariance(dataTrain, dataTest)

    # ======== 3. Train L21L1 model ==========
    clf = Class_L21L1(lambda1=1.55, lambda21=3.05)
    clf.fit(RsTrain, y_train)

    # ======== 4. Predict and evaluate ==========
    acc = clf.score(RsTest, y_test)
    print(f"Test accuracy: {acc:.4f}")

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
