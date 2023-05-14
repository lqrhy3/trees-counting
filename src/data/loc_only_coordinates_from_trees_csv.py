import os

import pandas as pd
from dotenv import load_dotenv


def main():
    df = pd.read_csv(os.environ['PATH_TO_RAW_TREE_LABELS'])
    loced_df = df.loc[:, ['latitude', 'longitude']]
    loced_df.to_csv(os.environ['PATH_TO_TREE_LABELS'], index=False)


if __name__ == '__main__':
    load_dotenv()
    main()
