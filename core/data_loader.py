import pandas as pd


class Data:
    def __init__(self):
        self.data_description = pd.read_csv("../Data/origination_feats.csv")
        self.data = pd.read_excel("../Data/morigination.xlsx", sheet_name="Sheet1")
