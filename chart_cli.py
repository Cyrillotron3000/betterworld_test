import argparse
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import pandas as pd

def plot_stock_data(signs, agg=None, start_date='2023-01-01', end_date=datetime.datetime.today().strftime('%Y-%m-%d')):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    for sign in signs.split(','):
        df = web.DataReader(sign, 'stooq', start=start_date, end=end_date)
        df.index = pd.to_datetime(df.index)

        if agg:
            df = df.resample(agg).mean()

        plt.plot(df['Close'], df['Volume'], color='r')
        plt.xlabel('Close')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes the relationship between stock prices and volumes")
    parser.add_argument("--signs", required=True, help="Comma-separated list of stock symbols")
    parser.add_argument("--agg", default=None, choices=['W', 'M', 'Y'], help="Aggregation level: W (week), M (month), Y (year)")
    parser.add_argument("--start", default='2023-01-01', help="Start date for collected data (format: YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.datetime.today().strftime('%Y-%m-%d'), help="End date for collected data (format: YYYY-MM-DD)")
    args = parser.parse_args()

    plot_stock_data(args.signs, args.agg, args.start, args.end)
