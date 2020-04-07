# standard library imports
import json
# third party imports
import requests

if __name__ == '__main__':
    url = 'http://127.0.0.1:5000/'
    data = '[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]'
    params = json.dumps({'query': data})
    response = requests.post(url=url, json=params)
    print(response.json())

