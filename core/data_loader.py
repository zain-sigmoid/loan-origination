import pandas as pd
import re


class Data:
    def __init__(self):
        print("loading data in data_loader")
        self.data_description = pd.read_csv("./data/origination_feats.csv")
        self.data = pd.read_excel("./data/small_data.xlsx", sheet_name="Sheet1")
        print("data loaded in data loader")

    @staticmethod
    def format_schema_from_excel(df) -> str:
        # Standardize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        schema_lines = []
        for _, row in df.iterrows():
            col = row.get("filed_name", "").strip()
            desc = row.get("description", "").strip()
            dtype = row.get("data_type", "").strip()
            values = row.get("values", "").strip()

            if values:
                values = re.sub(r"[\[\]']", "", values)  # Remove brackets/quotes
                parts = [v.strip() for v in values.split(",")]

                # Replace first hyphen with equal sign in each part
                values = []
                for part in parts:
                    values.append(
                        re.sub(r"^(\d+)\s*[-–]\s*", r"\1=", part)
                    )  # support hyphen and en dash

                values = ", ".join(values)
                desc += f" (values: {values})"

            if col:
                schema_lines.append(f"- {col}: {dtype}, {desc}")

        return "\n".join(schema_lines)
