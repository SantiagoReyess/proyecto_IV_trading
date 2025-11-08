import pandas as pd
from data_pre import clean_data
from stat_tests import prueba_adf


def main():

    #Download data for stocks and erase stocks with missing data (15 years historic)
    data = pd.read_csv("./data/stocks_data.csv", index_col=0, parse_dates=True)
    data = clean_data(data=data)

    #Filter non stationary series
    data, results, non_stationaries, stationaries = prueba_adf(data)

    #Select cointegrated pairs
    

    return 

if __name__ == "__main__":
    main()
