# standard library imports
import os
# third party imports
from flask_restful import Resource
import joblib
# local imports
from create_iris_model import CreateIrisModel
from iris_data_schemas import IrisQuerySchema


class IrisAPI(Resource):

    def __init__(self):
        if not os.path.exists(os.path.join('model_zoo', 'iris', 'iris_model.joblib')):
            CreateIrisModel.main()
        self.iris_model = joblib.load(os.path.join('model_zoo', 'iris', 'iris_model.joblib'))
        self.query_schema = IrisQuerySchema()

    def 