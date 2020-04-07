# Python Model Deployment

## Author
Nick Buker

## Description
Deploying a simple machine learning model with a restful flask app.

## Using this project
1. Clone this project to your local system
2. Create and activate a virtual environment using Python >= 3.6
3. Navigate to the root directory of this project and run the following command from the terminal
```bash
$ pip install -r requirements.txt
```
4. The model endpoint can be launched in debug mode using the following command
```bash
$ python app.py
```
**WARNING:** Do not deploy to production without editing line 38 of `app.py` to change debug to False.
```python
if __name__ == '__main__':
    app.run(debug=False)
```

## Project structure
```
├── model_bin/
│   └── lr.joblib
├── src/
│   ├── model_creation_pipeline.py
│   └── model_params.py
├── README.md
├── app.py
├── requirements.txt
└── test_query.py
```
- `model_bin/` - Stores trained model binary
- `src/` - Contains supporting Python scripts
    - `model_creation_pipeline.py` - Loads data, fits a model, and saved the trained model as a binary in the `model_bin/` directory
    - `model_params.py` - Specifies the parameters used for creating the model
- `app.py` - Launches flask app for the model API
- `requirements.txt` - Python packages required to run this project
- `test_query.py` - Sends a POST request containing input data to the API and receives a model output response
