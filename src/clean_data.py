# this file needs to be in the same directory as "p53_old_2010"
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("K8.data", header = None, low_memory = False)   # process the data with pandas
    shape = df.shape                                                 # check the shape of the data matrix
    print("Original number of rows:", shape[0])
    print("Original number of columns:", shape[1])

    print("Check original.head()", df.head())

    df = df.replace("?", np.nan)
    df[5408].replace(to_replace=["inactive", "active"], value=[0,1], inplace=True)                     # replace all "?" with np.nan
    df.drop(labels = [5409], axis = 1, inplace = True)              # drop the last column



    df = df.dropna(axis = 0)                                        # drop all rows with "NaN"
    print(df.head())

    shape = df.shape
    print("Cleaned number of rows:", shape[0])
    print("Cleaned number of columns:", shape[1])

    # output the cleaned dataset as "cleaned_K8.csv"
    # NOTE: when opening cleaned_K8.csv, use:
    #       pd.read_csv("cleaned_K8.csv", header = None, low_memory = False)
    df.to_csv("cleaned_K8.csv", sep = ",", index = False, header = False)

if __name__ == "__main__":
    main()
