# %%
import pandas as pd

# csv_path = "/home/julian/vlm_qa_data(0521).csv"
# csv_path = "/home/julian/vlm_qa_data_0521.csv"
# df = pd.read_csv(csv_path, encoding="big5")
# print(df.head())
xlsx_path = "/home/julian/vlm_qa_data.xlsx"
df = pd.read_excel(xlsx_path)

for index, row in df.iterrows():
    data_slice = row['data slice 名稱']
    dataset = row['dataset']
    print(f'"{data_slice}": "{dataset}",')

# %%
