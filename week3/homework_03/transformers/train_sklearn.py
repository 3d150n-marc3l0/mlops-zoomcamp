from typing import Tuple

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression



if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    print("Select categorical variables")
    categorical = ['PULocationID', 'DOLocationID']
    df = data[categorical]
    
    print("transform df to dict")
    data_dicts = df.to_dict(orient='records')
    
    print("Fit and transform a dict vectorizer from the data")
    dv = DictVectorizer()
    X = dv.fit_transform(data_dicts)

    y = data['duration'].values

    print("Fit a linear regression model with default parameters")
    model = LinearRegression()
    model.fit(X, y)

    print("Intercept:", model.intercept_)  # Print the intercept field
    
    return dv, model


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'