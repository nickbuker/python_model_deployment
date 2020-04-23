# third party imports
from marshmallow import Schema, fields
from marshmallow.validate import Range


gtz = 'Value must be greater than 0.'


class QuerySchema(Schema):
    ob_id = fields.Integer(required=True)
    sep_len = fields.Float(required=True, validate=Range(min=0, error=gtz))
    sep_wid = fields.Float(required=True, validate=Range(min=0, error=gtz))
    pet_len = fields.Float(required=True, validate=Range(min=0, error=gtz))
    pet_wid = fields.Float(required=True, validate=Range(min=0, error=gtz))


class OutputSchema(Schema):
    ob_id = fields.Integer(required=True)
    prediction = fields.Integer(required=True)
    probability = fields.Float(required=True, validate=Range(min=0, max=1, max_inclusive=True))
