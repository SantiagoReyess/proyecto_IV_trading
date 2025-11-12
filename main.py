import pandas as pd
from data_pre import clean_data
from stat_tests import prueba_adf
from stat_tests import johansen_cointegration_test
from stat_tests import encontrar_pares_cointegrados
from graphs import plot_normalized

def main():

    #Download data for stocks and erase stocks with missing data (15 years historic)
    data = pd.read_csv("./data/stocks_data.csv", index_col=0, parse_dates=True)
    data = clean_data(data=data)

    #Filter non stationary series
    #data, results, non_stationaries, stationaries = prueba_adf(data)

    #Select cointegrated pairs
    #pares, stats = encontrar_pares_cointegrados(data, alpha=0.05)

    #Filter only the pair selected
    data = data.loc[:, ["AMD", "TSM"]]

    # Plot stocks to verify cointegration
    #plot_normalized(series1=stock1, series2=stock2)
 
    # Train, Test
    split = int(len(data) * .60)
    train = data.iloc[:split]
    test = data.iloc[split:]

    # Start Backtesting


    return print(len(data), len(train), len(test))

if __name__ == "__main__":
    main()
