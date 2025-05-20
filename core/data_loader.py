import pandas as pd
import re
import yaml
import os
import numpy as np


class Data:
    def __init__(self):
        print("loading data in data_loader")
        self.data_description = pd.read_csv("./data/origination_feats_up.csv")
        # excel_data = pd.read_excel("./data/morigination.xlsx", sheet_name="Sheet1")
        excel_data = pd.read_parquet("./data/origination.parquet")
        print("sanitizing")
        # import pdb
        # pdb.set_trace()
        self.data = self.sanitize_data(excel_data)
        self.data_sample = self.data.sample(10, random_state=42)
        # print(self.data.dtypes)
        # print(self.data["county_code"].unique())
        # self.data = pd.read_parquet("./data/origination.parquet")
        print("data loaded in data loader")

    @staticmethod
    def format_schema_from_excel(df) -> str:
        # Standardize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        schema_lines = []
        for _, row in df.iterrows():
            col = row.get("field_name", "").strip()
            desc = row.get("description", "").strip()
            dtype = row.get("data_type", "").strip()
            values = row.get("values", "").strip()

            if values:
                values = re.sub(r"[\[\]]", "", values)  # Remove brackets/quotes
                parts = [v.strip() for v in values.split(",")]

                formatted_values = []
                for part in parts:
                    match = re.match(r"['\"]?(\d+)['\"]?\s*[-–]\s*(.+)", part)
                    if match:
                        key, label = match.groups()
                        formatted_values.append(f"'{key}'={label.strip()}")
                    else:
                        formatted_values.append(
                            part
                        )  # fallback if format doesn't match

                desc += f" (values: {', '.join(formatted_values)})"

            if col:
                schema_lines.append(f"- {col}: {dtype}, {desc}")

        return "\n".join(schema_lines)

    def safe_convert(self, val):
        if pd.isna(val):
            return np.nan
        elif val == int(val):
            return str(int(val))
        else:
            return str(val)

    def sanitize_data(self, df):
        config_path = os.path.join(os.path.dirname(__file__), "config.yml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        cols_to_string = config["cols_to_string"]
        data = df.copy()
        for col in cols_to_string:
            try:
                if col != "county_code":
                    data[col] = data[col].astype("string")
                else:
                    data["county_code"] = data["county_code"].apply(self.safe_convert)
            except Exception as e:
                print(f"Failed to convert {col}: {e}")
        return data
