# standard library imports
import json
import os
from typing import Dict, List, Tuple
# third party imports
from flask import Response, abort, request
from flask_restful import Resource
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
# local imports
from model_zoo.iris.create_iris_model import CreateIrisModel
from model_zoo.iris.iris_data_schemas import IrisQuerySchema


class IrisAPI(Resource):

    def __init__(self):
        self.iris_model = self._load_model()
        self.query_schema = IrisQuerySchema()

    def post(self) -> Response:
        """

        Returns
        -------

        """
        data = json.loads(request.json)
        self._validate_data(data=data)
        output = self._generate_output(*self._prep_data_for_model(data))
        return Response(json.dumps(output), mimetype='application/json')

    def _load_model(self) -> LogisticRegression:
        """

        Returns
        -------

        """
        model_path = os.path.join('model_zoo', 'iris', 'iris_model.joblib')
        if not os.path.exists(model_path):
            CreateIrisModel.main()
        return joblib.load(model_path)

    def _validate_data(self, data: List[Dict]) -> None:
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        errors = self.query_schema.validate(data=data, many=True)
        if errors:
            abort(400, str(errors))
        return

    def _prep_data_for_model(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        data_lists = []
        for d in data:
            data_lists.append([d['ob_id'], d['sep_len'], d['sep_wid'], d['pet_len'], d['pet_wid']])
        data_df = pd.DataFrame(data=data_lists, columns=['ob_id', 'sep_len', 'sep_wid', 'pet_len', 'pet_wid'])
        return data_df.loc[:, ['sep_len', 'sep_wid', 'pet_len', 'pet_wid']], data_df.loc[:, 'ob_id']

    def _generate_output(self, mod_data: pd.DataFrame, ob_ids: pd.Series) -> List[Dict]:
        """

        Parameters
        ----------
        mod_data
        ob_ids

        Returns
        -------

        """
        preds = self.iris_model.predict(mod_data).tolist()
        probs = self.iris_model.predict_proba(mod_data).max(axis=1).tolist()
        output = []
        for i, ob_id in enumerate(ob_ids):
            output.append({
                'ob_id': ob_id,
                'prediction': preds[i],
                'probability': probs[i]
            })
        return output
