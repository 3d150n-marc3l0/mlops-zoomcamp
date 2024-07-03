import pandas as pd
import batch
from datetime import datetime
from deepdiff import DeepDiff


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']

    pre_df = batch.prepare_data(df, categorical)

    expected_df = pd.DataFrame(
        [["-1", "-1", dt(1, 1),    dt(1, 10), 9.0],
         [ "1",  "1", dt(1, 2),    dt(1, 10), 8.0]],
        columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration'])


    diff = DeepDiff( pre_df.to_dict(), expected_df.to_dict(), significant_digits=1 ,ignore_order=True)
    
    assert 'type_changes' not in diff
    assert 'values_changed' not in diff 
    assert len(pre_df) == 2