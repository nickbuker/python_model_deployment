# standard library imports
import os
import sys
from typing import Tuple
# third party imports
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# local imports
sys.path.append(os.path.join('.'))
from src.model_params import model_params


class ModelCreationPipeline:

    @staticmethod
    def main() -> None:
        X, y = ModelCreationPipeline._load_data()
        lr = ModelCreationPipeline._train_model(X, y)
        ModelCreationPipeline._save_model(lr)

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
            mod_path=os.path.join('model_bin', 'lr.joblib')
    ) -> None:
        """

        Parameters
        ----------
        lr

        Returns
        -------
        None
        """
        joblib.dump(value=lr, filename=mod_path)
        return


if __name__ == '__main__':
    ModelCreationPipeline.main()
