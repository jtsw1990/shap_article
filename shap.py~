#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


output_path = os.path.join(os.getcwd(), "output")


df = pd.read_csv("insurance.csv")

rating_factors =  list(df.columns[:-1])
claims = df.columns[-1]

def plot_export_corr(df, x, y, output_path):
    plt.scatter(x=df[x], y=df[y])
    plt.savefig(output_path + "\\{}_corr.png".format(x))
    plt.clf()

rating_factors_encoded = pd.get_dummies(df[rating_factors])

lm = LinearRegression()
lm.fit(rating_factors_encoded, df[claims])

y_pred = lm.predict(rating_factors_encoded)

print(df[claims], y_pred)


if __name__ == "__main__":
    for factor in rating_factors:
        plot_export_corr(df, factor, claims, output_path)
