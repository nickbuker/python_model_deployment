# standard library imports
import json
# third party imports
from marshmallow import ValidationError
import requests
# local imports
from model_zoo.iris.iris_data_schemas import IrisOutputSchema

data = [
    {
        'ob_id': 3,
        'sep_len': 4.6,
        'sep_wid': 3.1,
        'pet_len': 1.5,
        'pet_wid': 0.2,
    },
    {
        'ob_id': 38,
        'sep_len': 4.4,
        'sep_wid': 3.0,
        'pet_len': 1.3,
        'pet_wid': 0.2,
    },
    {
        'ob_id': 53,
        'sep_len': 5.5,
        'sep_wid': 2.3,
        'pet_len': 4.0,
        'pet_wid': 1.3,
    },
    {
        'ob_id': 78,
        'sep_len': 6.0,
        'sep_wid': 2.9,
        'pet_len': 4.5,
        'pet_wid': 1.5,
    },
    {
        'ob_id': 123,
        'sep_len': 6.3,
        'sep_wid': 2.7,
        'pet_len': 4.9,
        'pet_wid': 1.8,
    },
    {
        'ob_id': 141,
        'sep_len': 6.9,
        'sep_wid': 3.1,
        'pet_len': 5.1,
        'pet_wid': 2.3,
    },
]


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000/'
    json_data = json.dumps(data)
    response = requests.post(url=url, json=json_data)
    if response.status_code == 200:
        response_json = response.json()
        val_errors = IrisOutputSchema().validate(data=response_json, many=True)
        if val_errors:
            raise ValidationError(f'Following response validation errors occurred: {val_errors}')
        print(response_json)
    else:
        print(response.status_code, response.reason)
        print(response.json())
