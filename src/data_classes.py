from marshmallow import Schema, fields


class ModelQuery(Schema):
    query = fields.List(fields.List(fields.Float()), required=True)


class ModelOutput(Schema):
    prediction = fields.List(fields.Integer(), required=True)
    probability = fields.List(fields.Float(), required=True)
