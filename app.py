# standard library imports
from argparse import ArgumentParser
# third party imports
from flask import Flask
from flask_restful import Api
# local imports
from model_zoo.iris.iris_api import IrisAPI


app = Flask(__name__)
api = Api(app)
api.add_resource(IrisAPI, '/iris/')


if __name__ == '__main__':
    debug = False
    arg_parser = ArgumentParser(allow_abbrev=False)
    arg_parser.add_argument(
        '-d',
        '--debug',
        help='Optional arg indicating app debug mode.',
        action='store_true',
    )

    args = arg_parser.parse_args()
    if args.debug:
        debug = True
    app.run(debug=debug)
