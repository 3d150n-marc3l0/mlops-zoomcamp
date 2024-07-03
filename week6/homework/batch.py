#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
from datetime import datetime

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename, options):
    df = pd.read_parquet(filename, storage_options=options)


def prepare_data(df: pd.DataFrame, categorical: list):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def save_data(df: pd.DataFrame, output_file: str):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    
    )

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def main(year: int = 2024, month: int = 12):
    #input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    print(input_file)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    #df = read_data(input_file, options)
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    print(f"df: {len(df)}")
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.sum())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred


    #df_result.to_parquet(output_file, engine='pyarrow', index=False)
    save_data(df_result, output_file)

    



if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    print(f"year:{year}, month:{month}")
    main(year, month)