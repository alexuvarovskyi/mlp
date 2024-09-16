import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import os


def generate_sample_dataframe(rows=1_000_000, cols=50):
    np.random.seed(0)
    data = np.random.randn(rows, cols)
    columns = [f'col_{i}' for i in range(cols)]
    df = pd.DataFrame(data, columns=columns)
    return df

def save_csv(df, filename):
    df.to_csv(filename, index=False)

def load_csv(filename):
    return pd.read_csv(filename)

def save_excel(df, filename):
    df.to_excel(filename, index=False)

def load_excel(filename):
    return pd.read_excel(filename)

def save_parquet(df, filename):
    df.to_parquet(filename)

def load_parquet(filename):
    return pd.read_parquet(filename)

def save_hdf(df, filename):
    df.to_hdf(filename, key='df', mode='w')

def load_hdf(filename):
    return pd.read_hdf(filename, key='df')

def save_feather(df, filename):
    df.to_feather(filename)

def load_feather(filename):
    return pd.read_feather(filename)

def save_npy(df, filename):
    np.save(filename, df.to_numpy())

def load_npy(filename):
    data = np.load(filename)
    return pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])

def measure_time(func, *args):
    start_time = time.perf_counter()
    func(*args)
    end_time = time.perf_counter()
    return end_time - start_time

def benchmark_formats(df):
    formats = {
        "CSV": ("data.csv", save_csv, load_csv),
        "Excel": ("data.xlsx", save_excel, load_excel),
        "Parquet": ("data.parquet", save_parquet, load_parquet),
        "HDF5": ("data.h5", save_hdf, load_hdf),
        "Feather": ("data.feather", save_feather, load_feather),
        "NPY": ("data.npy", save_npy, load_npy),
    }

    results = []

    for fmt, (filename, save_func, load_func) in tqdm(formats.items()):
        save_time = measure_time(save_func, df, filename)
        load_time = measure_time(load_func, filename)
        file_size =  os.path.getsize(filename) / (1024**2)
        results.append((fmt, save_time, load_time, file_size))
    
    print(f"{'Format':<10} {'Save Time (s)':<15} {'Load Time (s)':<15}" 'File size')
    print("-" * 50)
    for fmt, save_time, load_time, file_size in results:
        print(f"{fmt:<10} {save_time:<15.5f} {load_time:<15.5f} {file_size:.2f} MB")

if __name__ == "__main__":
    print(f"Generating Sample DataFrame... in {measure_time(generate_sample_dataframe):.5f} seconds")
    df = generate_sample_dataframe()
    benchmark_formats(df)