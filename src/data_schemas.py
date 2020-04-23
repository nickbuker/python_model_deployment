# third party imports
from marshmallow import Schema, fields
from marshmallow.validate import Range


class QuerySchema(Schema):
    query = fields.List(fields.List(fields.Float()), required=True)


class OutputSchema(Schema):
    ob_id = fields.Integer(required=True)
    prediction = fields.Integer(required=True)
    probability = fields.Float(required=True, validate=Range(min=0, max=1, max_inclusive=True))
