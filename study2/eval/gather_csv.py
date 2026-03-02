# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from study2.eval.process import DataPoint, TorchPolicy
from study2.static import *



data = DataPoint.load_all_processed_data()
cond0_data = data['cond0']
cond1_data = data['cond1']
all_data = data['all']


def gather(flat=True):
    print(f"Number of cond0 data points: {len(cond0_data)}")
    print(f"Number of cond1 data points: {len(cond1_data)}")
    print(f"Number of all data points: {len(all_data)}")

    data_df = cond0_data[0].to_pandas_flat() if flat else cond0_data[0].to_pandas()

    for dp in cond0_data[1:]:
        # append to dataframe
        new_df = dp.to_pandas_flat() if flat else dp.to_pandas()
        data_df = pd.concat([data_df, new_df], ignore_index=True)

    for dp in cond1_data:
        new_df = dp.to_pandas_flat()  if flat else dp.to_pandas()
        data_df = pd.concat([data_df, new_df], ignore_index=True)

    # Save to CSV
    # print(data_df)
    return data_df


def main():
    """Look at all processed data and gather into a single CSV file for analysis."""
    flat = True
    data_df = gather(flat=flat)
    fname = 'processed_data_summary_flat.csv' if flat else 'processed_data_summary_stacked.csv'
    data_df.to_csv(fname, index=False)


    flat = False
    data_df = gather(flat=flat)
    fname = 'processed_data_summary_flat.csv' if flat else 'processed_data_summary_stacked.csv'
    data_df.to_csv(fname, index=False)






if __name__ == "__main__":
    main()
