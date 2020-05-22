# local imports
from model_zoo.iris.iris_data_schemas import gtz_error, IrisOutputSchema, IrisQuerySchema


query_schema = IrisQuerySchema()
output_schema = IrisOutputSchema()


nvi = ['Not a valid integer.']
nvn = ['Not a valid number.']
gtz = [gtz_error]


def test_gtz_error():
    assert gtz_error == 'Value must be greater than 0.'


def test_iris_query_schema_ob_id_type():
    data = {
        'ob_id': 1.0,
        'sep_len': 1.0,
        'sep_wid': 1.0,
        'pet_len': 1.0,
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data, many=False)
    assert errors['ob_id'] == nvi


def test_iris_query_schema_sep_len_type():
    data = {
        'ob_id': 1,
        'sep_len': 'abc',
        'sep_wid': 1.0,
        'pet_len': 1.0,
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['sep_len'] == nvn


def test_iris_query_schema_sep_len_range():
    data = {
        'ob_id': 1,
        'sep_len': -1.0,
        'sep_wid': 1.0,
        'pet_len': 1.0,
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['sep_len'] == gtz


def test_iris_query_schema_sep_wid_type():
    data = {
        'ob_id': 1,
        'sep_len': 1.0,
        'sep_wid': 'abc',
        'pet_len': 1.0,
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['sep_wid'] == nvn


def test_iris_query_schema_sep_wid_range():
    data = {
        'ob_id': 1,
        'sep_len': 1.0,
        'sep_wid': -1.0,
        'pet_len': 1.0,
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['sep_wid'] == gtz


def test_iris_query_schema_pet_len_type():
    data = {
        'ob_id': 1,
        'sep_len': 1.0,
        'sep_wid': 1.0,
        'pet_len': 'abc',
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['pet_len'] == nvn


def test_iris_query_schema_pet_len_range():
    data = {
        'ob_id': 1,
        'sep_len': 1.0,
        'sep_wid': 1.0,
        'pet_len': -1.0,
        'pet_wid': 1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['pet_len'] == gtz


def test_iris_query_schema_pet_wid_type():
    data = {
        'ob_id': 1,
        'sep_len': 1.0,
        'sep_wid': 1.0,
        'pet_len': 1.0,
        'pet_wid': 'abc',
    }
    errors = query_schema.validate(data=data)
    assert errors['pet_wid'] == nvn


def test_iris_query_schema_pet_wid_range():
    data = {
        'ob_id': 1,
        'sep_len': 1.0,
        'sep_wid': 1.0,
        'pet_len': 1.0,
        'pet_wid': -1.0,
    }
    errors = query_schema.validate(data=data)
    assert errors['pet_wid'] == gtz


def test_iris_output_schema_ob_id_type():
    data = {
        'ob_id': 1.0,
        'prediction': 1,
        'probability': 1.0,
    }
    errors = output_schema.validate(data=data)
    assert errors['ob_id'] == nvi


def test_iris_output_schema_prediction_type():
    data = {
        'ob_id': 1,
        'prediction': 1.0,
        'probability': 1.0,
    }
    errors = output_schema.validate(data=data)
    assert errors['prediction'] == nvi


def test_iris_output_schema_probability_type():
    data = {
        'ob_id': 1,
        'prediction': 1.0,
        'probability': 'abc',
    }
    errors = output_schema.validate(data=data)
    assert errors['probability'] == nvn


def test_iris_output_schema_probability_min():
    data = {
        'ob_id': 1,
        'prediction': 1,
        'probability': -1.0,
    }
    errors = output_schema.validate(data=data)
    assert errors['probability'] == ['Must be greater than or equal to 0 and less than or equal to 1.']


def test_iris_output_schema_probability_max():
    data = {
        'ob_id': 1,
        'prediction': 1,
        'probability': 2.0,
    }
    errors = output_schema.validate(data=data)
    assert errors['probability'] == ['Must be greater than or equal to 0 and less than or equal to 1.']
