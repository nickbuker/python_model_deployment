# third party imports
from sklearn.linear_model import LogisticRegression
# local imports
from model_zoo.iris.iris_model_params import model_params


def test_iris_model_params_dict():
    assert isinstance(model_params, dict)


def test_iris_model_params_ingestion():
    # test the model_params can be ingested by LogisticRegression
    LogisticRegression(**model_params)
