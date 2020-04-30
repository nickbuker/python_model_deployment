# standard library imports
import os
from typing import Tuple
# third party imports
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# local imports
from model_zoo.iris.iris_model_params import model_params


class CreateIrisModel:

    @staticmethod
    def main() -> None:
        """Loads data, trains, and saves sklearn logistic regression model

        Returns
        -------
        None
        """
        X, y = CreateIrisModel._load_data()
        lr = CreateIrisModel._train_model(X, y)
        CreateIrisModel._save_model(lr)

    @staticmethod
    def _load_data() -> Tuple[np.ndarray, np.ndarray]:
        """Loads iris dataset and generates input and target data

        Returns
        -------
        Tuple
            input data
            target data
        """
        iris = load_iris()
        X = iris.data
        y = iris.target
        return X, y

    @staticmethod
    def _train_model(
            X: np.ndarray,
            y: np.ndarray
    ) -> LogisticRegression:
        """Trains a logistic regression model using the iris data and model params

        Parameters
        ----------
        X
            input data
        y
            target data

        Returns
        -------
        LogisticRegression
            trained sklearn logistic regression model
        """
        lr = LogisticRegression(**model_params)
        lr.fit(X=X, y=y)
        return lr

    @staticmethod
    def _save_model(
            lr: LogisticRegression,
            mod_path=os.path.join('model_zoo', 'iris', 'iris_model.joblib')
    ) -> None:
        """Saves trained model binary in joblib format

        Parameters
        ----------
        lr
            trained sklearn logistic regression model

        Returns
        -------
        None
        """
        joblib.dump(value=lr, filename=mod_path)
        return


if __name__ == '__main__':
    CreateIrisModel.main()
