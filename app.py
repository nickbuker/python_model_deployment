# standard library imports
import json
import os
# third party imports
from flask import Flask, request, Response
from flask_restful import Api, reqparse, Resource
import joblib
from marshmallow import ValidationError
import numpy as np
import pandas as pd
# local imports
from src.data_schemas import QuerySchema


app = Flask(__name__)
api = Api(app)


# load trained model
lr = joblib.load(os.path.join('model_bin', 'lr.joblib'))

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class IrisProb(Resource):

    def post(self):
        # try:
        #     data = QuerySchema().load(request.get_json())
        # except ValidationError as e:
        #     print(e)
        data = json.loads(request.get_json())
        data_lists = []
        for d in data:
            data_lists.append([d['ob_id'], d['sep_len'], d['sep_wid'], d['pet_len'], d['pet_wid']])
        data_df = pd.DataFrame(data=data_lists, columns=['ob_id', 'sep_len', 'sep_wid', 'pet_len', 'pet_wid'])
        X = data_df.loc[:, ['sep_len', 'sep_wid', 'pet_len', 'pet_wid']]
        preds = lr.predict(X).tolist()
        probs = lr.predict_proba(X).max(axis=1).tolist()
        output = []
        for i, ob_id in enumerate(data_df['ob_id']):
            output.append({
                'ob_id': ob_id,
                'prediction': preds[i],
                'probability': probs[i]
            })
        return Response(json.dumps(output), mimetype='application/json')


api.add_resource(IrisProb, '/')


if __name__ == '__main__':
    app.run(debug=True)
