import pandas as pd


class Data:
    def __init__(self):
        print("loading data in data_loader")
        self.data_description = pd.read_csv("./data/origination_feats.csv")
        self.data = pd.read_excel("./data/small_data.xlsx", sheet_name="Sheet1")
        print("data loaded in data loader")
