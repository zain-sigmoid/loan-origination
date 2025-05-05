import pandas as pd


class Data:
    def __init__(self):
        self.data_description = pd.read_csv("./data/origination_feats.csv")
        self.data = pd.read_excel("./data/morigination.xlsx", sheet_name="Sheet1")
        print(self.data.shape)
