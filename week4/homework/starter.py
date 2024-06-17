import pickle
import pandas as pd
import argparse
import os

DATASET_DIR = "Data"
OUTPUT_DIR = "output"
DATA_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data'
categorical = ['PULocationID', 'DOLocationID']
DEFAULT_MODEL_FILE = 'model.bin'

for dir in [DATASET_DIR, OUTPUT_DIR]:
    if os.path.isdir(dir): 
        print(f"The {dir} directory exists")
        continue
    # if the directory is  
    # not present then create it. 
    os.makedirs(dir, exist_ok=True)
    print(f"The {dir} directory is created")


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def save_results(df, y_pred, output_file):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = df[['ride_id']].copy()
    df_result['prediction'] = y_pred
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def load_model(model_file: str):
    print(f'Loading model {model_file}...')
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model



def apply_model(model_file, input_file:str, output_file:str):
    print(f'reading data {input_file}...')
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    
    print('predicting...')
    dv, model = load_model(model_file)
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(f'the mean of prediction is {y_pred.mean()}')
    
    print(f'save results {output_file}...')
    save_results(df, y_pred, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ride duration prediction.')
    parser.add_argument('taxi_type', type=str, help='enter taxi type', default='yellow')
    parser.add_argument('year', type=int, help='enter year from 2023')
    parser.add_argument('month', type=int,  help='enter month from 1 to 12')
    args = parser.parse_args()

    taxi_type = args.taxi_type
    year = args.year
    month = args.month

    path = "./"
    dirs = os.listdir( path )
    for file in dirs:
        print(file)

    model_file = DEFAULT_MODEL_FILE
    input_file = f'{DATA_URL}/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'{OUTPUT_DIR}/pred_yellow_tripdata_{year:04}-{month:02}.parquet'
    
    apply_model(model_file, input_file, output_file)
