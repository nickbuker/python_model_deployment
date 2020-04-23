# standard library imports
import json
# third party imports
from marshmallow import ValidationError
import requests
# local imports
from src.data_schemas import OutputSchema


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000/'
    data = '[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]'
    params = json.dumps({'query': data})
    response = requests.post(url=url, json=params)
    val_errors = OutputSchema().validate(data=response.json(), many=True)
    if val_errors:
        raise ValidationError(f'Following response validation errors occurred: {val_errors}')
    print(response.json())
