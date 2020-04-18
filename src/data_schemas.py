from marshmallow import Schema, fields


class QuerySchema(Schema):
    query = fields.List(fields.List(fields.Float()), required=True)


class OutputSchema(Schema):
    prediction = fields.List(fields.Integer(), required=True)
    probability = fields.List(fields.Float(), required=True)
