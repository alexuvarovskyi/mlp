## Pandas Formats Profiling


We benchmarked 6 different file formats for storing pandas dataframes. The formats are: CSV, Parquet, HDF5, Feather, NPY, and Excel.


| Format   | Save Time (s) | Load Time (s) | File size |
|----------|---------------|---------------|-----------|
| CSV      | 32.19560      | 5.61545       | 936.09 MB |
| Excel    | 512.45685     | 270.20350     | 622.33 MB |
| Parquet  | 1.56897       | 0.14528       | 394.84 MB |
| HDF5     | 0.12157       | 0.38918       | 389.11 MB |
| Feather  | 0.36712       | 0.11358       | 381.57 MB |
| NPY      | 0.05290       | 0.05778       | 381.47 MB |


We tested a tubular dataset with 100,000 rows and 50 columns in np.float. 
Based on the results, the best format for saving and loading pandas dataframes is NPY. It is the fastest and has the smallest file size.
