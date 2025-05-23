import pandas as pd
import re
import yaml
import os
import numpy as np
from core.config_p import categorical_columns_to_map


class Data:
    def __init__(self):
        print("loading data in data_loader")
        self.data_description_prev = pd.read_csv("./data/origination_feats_up.csv")
        self.data_description = pd.read_csv("./data/data_description_latest.csv")
        # excel_data = pd.read_excel("./data/morigination.xlsx", sheet_name="Sheet1")
        excel_data = pd.read_parquet("./data/origination.parquet")
        print("sanitizing")
        # import pdb
        # pdb.set_trace()
        self.data = self.sanitize_data(excel_data)
        self.data = self.generate_mapping(self.data, self.data_description_prev)
        print("getting final columns")
        self.data = self.get_final_columns(self.data)
        self.data_sample = self.data.sample(10, random_state=42)
        # print(self.data["total_units"].unique(), self.data["total_units"].dtype)
        print("data loaded in data loader")

    def get_final_columns(self, data):
        data_c = data.copy()
        ethnicity_columns = [
            col for col in data_c.columns if "ethnicity" in col.lower()
        ]
        race_columns = [col for col in data_c.columns if "race" in col.lower()]
        sex_columns = [col for col in data_c.columns if "sex" in col.lower()]
        denial_reason_columns = [
            col for col in data_c.columns if "denial_reason" in col.lower()
        ]
        sorted_denial_reason_columns = sorted(
            denial_reason_columns, key=lambda x: int(x.split("-")[-1])
        )

        data_c["applicant_ethnicity"] = data_c["derived_ethnicity"]
        data_c["applicant_race"] = data_c["derived_race"]
        data_c["applicant_sex"] = data_c["derived_sex"]
        data_c["denial_reason"] = (
            data_c[sorted_denial_reason_columns].bfill(axis=1).iloc[:, 0]
        )
        data_c.drop(
            columns=ethnicity_columns
            + race_columns
            + sex_columns
            + denial_reason_columns,
            inplace=True,
        )

        return data_c

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
            # return pd.NA
            return "Not Available"
        elif isinstance(val, float) and val.is_integer():
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
                # if col != "county_code":
                #     data[col] = data[col].astype("string")
                # else:
                #     data["county_code"] = data["county_code"].apply(self.safe_convert)
                data[col] = data[col].apply(self.safe_convert)
            except Exception as e:
                print(f"Failed to convert {col}: {e}")
        return data

    def parse_value_string(self, value_str):
        mapping = {}
        if pd.isna(value_str):
            return mapping
        # Matches numeric, alphabetic, or alphanumeric codes like '1', 'A', '1A', etc.
        pattern = re.findall(r"'?([\w\-\.]+)'?\s*[-–]\s*([^,'\n]+)", value_str)
        for code, desc in pattern:
            mapping[str(code).strip()] = desc.strip()
        return mapping

    def safe_upper(self, val):
        if pd.isna(val):
            return pd.NA
        return str(val).upper()

    def generate_mapping(self, data, data_description):

        data_description_cat = data_description[
            data_description["Field Name"].isin(categorical_columns_to_map)
        ]
        #### create a mapping dictionary####
        column_value_dict = {}
        for _, row in data_description_cat.iterrows():
            column_name = row["Field Name"].strip()
            value_str = row["Values"]
            parsed_values = self.parse_value_string(value_str)
            if parsed_values:
                column_value_dict[column_name] = parsed_values

        column_value_dict["conforming_loan_limit"] = {
            "C": "Conforming",
            "NC": "Nonconforming",
            "U": "Undetermined",
            "NA": "Not Applicable",
        }

        ### conver the values in the categorical column of data to categories ###
        for col, mapping in column_value_dict.items():
            if col in data.columns:
                # Ensure all values are strings for safe replacement
                data[col] = data[col].astype(str).map(mapping).fillna(data[col])
                data[col] = data[col].apply(self.safe_upper)

        return data
