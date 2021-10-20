import numpy as np
import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

df_excel_stock = pd.read_csv('US STOCK TICKERS.csv')
df_excel_etf = pd.read_csv('US ETF TICKERS.csv')
df_excel_crypto = pd.read_csv('TOP 100 CRYPTO TICKERS csv.csv')
df_concat = pd.concat((df_excel_stock, df_excel_etf, df_excel_crypto), ignore_index=True)
df_concat.drop(columns=df_concat.iloc[:, 2:], inplace=True, axis=1)
# st.write(df_concat)
default_list1 = ["AAPL | Apple Inc. Common Stock", "DB | Deutsche Bank AG Common Stock"]
default_list2 = ["GLD | SPDR Gold Trust", "SPY | SPDR S&P 500", "AGG | iShares Core U.S. Aggregate Bond ETF"]
default_list3 = []
# "BTC-USD | Bitcoin"
NUM_TRADING_DAYS = 252
START_CAPITAL = 1000
compared_stock = "SPY"
sel = []
STOCK_LIST = []
st.header("Portfolio allocation by Modern Portfolio Theory: Monte Carlo Simulation")
with st.container():
    st.write("Welcome to my Portfolio Simulator Web App! This is an app that estimates optimal portfolio using"
             " Monte Carlo Simulations, it aims to achieve the highest expected return for a given amount of risk"
             " and also finds the lowest risk portfolio. ")
    st.write("Disclaimer: This is only for data visualisation purposes only, do your own due diligence when investing."
             " Past returns do not guarantee future returns!")


def get_dataset():
    # get data from yahoo finance and store it in a pandas dataframe
    stock_data = {}

    for stock in STOCK_LIST:
        stock_data[stock] = yf.Ticker(stock).history(stock, start=START_DATE, end=END_DATE)['Close']

    return pd.DataFrame(stock_data)


def set_new_date_for_available_data():
    # st.write(dataset)
    # get the series of dates where NaN value ends:
    # (series of dates when the stock has just been listed on exchange or equals to start_date variable)
    listing_date = dataset.apply(pd.Series.first_valid_index)

    # debug:
    # st.write(f"Initial listing date:\n{listing_date}")

    global START_DATE
    if listing_date.eq(START_DATE).all():
        pass

    else:
        # get the series containing info containing the latest listed ticker and its latest available date
        latest_listed_stock_date = listing_date.nlargest(1)
        # print(latest_listed_stock_date.index[0])

        # set the new starting date such that there are no NaN values in dataframe
        START_DATE = latest_listed_stock_date[0].date()

        st.warning(f"Yahoo Finance data not available for ticker \"{latest_listed_stock_date.index[0]}\" "
                   f"before {START_DATE},"f" new start date is set to {START_DATE}")

        return latest_listed_stock_date.index[0]


def plot_dataset(dataset_):
    st.write("***")
    st.subheader("Graph of asset prices/USD against time")
    st.line_chart(dataset_)


def get_normalised_daily_return(dataset_):
    # calculate the normalized daily returns:  r = log(Y(t+1) / Y(t))
    # need to measure all returns in comparable metrics
    log_daily_return_ = np.log(dataset_/dataset_.shift(1))

    st.subheader("Graph of normalised (logged) % daily returns")
    st.line_chart(log_daily_return_)
    st.write("***")

    return log_daily_return_[1:]


def annualised_mean_covariance(log_daily_return_):
    # calculating the annual mean and covariance
    mean_return_annual_ = log_daily_return_.mean() * NUM_TRADING_DAYS
    cov_matrix_annual_ = log_daily_return_.cov() * NUM_TRADING_DAYS
    return mean_return_annual_, cov_matrix_annual_


def display_mean_covariance_table():
    col1, col2 = st.columns((1.1, 3))
    with col1:
        st.subheader("Mean of annual returns:")

        def pct_round(x):
            return ("%.1f" % (x * 100)) + "%"

        mean_return_annual_pct = mean_return_annual.apply(pct_round)
        mean_return_annual_pct.name = "Return"

        st.write(mean_return_annual_pct)

    with col2:
        st.subheader("")
        st.subheader("Covariance of annual returns:")
        st.subheader("")
        st.table(cov_matrix_annual)

    for k in mean_return_annual:
        st.write()
        if str(k) == "nan":
            st.error("Ticker may have been delisted.")
            st.stop()
        else:
            pass


def correlation_heatmap(log_daily_return_):
    st.subheader("")
    st.subheader("Correlation matrix of underlying assets")
    correlation_matrix = log_daily_return_.corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap=sns.color_palette("Blues", 100), fmt='.2f')
    st.write(fig)
    st.write("***")


def generate_random_portfolios():
    # generate n sets of random variables, index in array corresponds to the nth random portfolio generated.
    portfolio_mean = []
    portfolio_sd = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        # generate random weights which sum = 1 for each portfolio:
        w = np.random.random(len(STOCK_LIST))
        w = w / np.sum(w)
        portfolio_weights.append(w)
        # portfolio mean:
        # summation of the product of (weights and mean annual return of individual asset)
        portfolio_mean.append(np.sum(w * mean_return_annual))
        # portfolio standard deviation:
        # square root of ( dot product of transposed weight matrix, (covariance matrix, weight matrix) )
        portfolio_sd.append(np.sqrt(np.dot(w.T, np.dot(cov_matrix_annual, w))))

    return np.array(portfolio_weights), np.array(portfolio_mean), np.array(portfolio_sd)


def calculate_portfolio_max_sharpe_min_risk(p_mean, p_stddev):
    # This is a function that retrieves the statistics array and weight array of the 2 optimal portfolios
    # Create a new dataframe with columns p_mean, p_stddev and p_sharpeRatio:
    df = pd.DataFrame({'p_means': p_mean, 'p_stdDevs': p_stddev})
    df['Sharpe_ratio'] = df['p_means'] / df['p_stdDevs']

    # debug print:
    # print(df)

    # Extracting the dataframe row containing index of portfolio with max sharpe ratio
    df_max_sharpe = df[df['Sharpe_ratio'] == df['Sharpe_ratio'].max()]
    # retrieve the index from single row df_max_sharpe to get corresponding row of the 2D weights array
    index = df_max_sharpe.index
    q = []
    for index in index:
        q.append(index)
    b = q[0]
    max_sharpe_weights = p_weights[int(b)]

    # Extracting the dataframe row containing index of portfolio with lowest risk (SD)
    df_min_risk = df[df['p_stdDevs'] == df['p_stdDevs'].min()]
    # retrieve the index from single row df_min_risk to get corresponding row of the 2D weights array
    index = df_min_risk.index
    c = []
    for index in index:
        c.append(index)
    d = c[0]
    min_risk_weights = p_weights[int(d)]

    return max_sharpe_weights, np.array(df_max_sharpe), min_risk_weights, np.array(df_min_risk)


def scatter_plot_optimal_portfolios(p_mean, p_sd):
    st.subheader('Monte Carlo Simulation')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Function the does a scatter plot of the randomly generated portfolios, including the two optimal portfolios.
    # x-axis = portfolio SD and y-axis = portfolio mean
    plt.figure(figsize=(10, 6))
    plt.scatter(p_sd, p_mean, c=p_mean/p_sd, cmap='Spectral', marker='o', edgecolors='k')
    plt.style.use('seaborn')
    plt.grid(True)
    plt.xlabel("Expected Volatility (SD)")
    plt.ylabel("Expected Return (Mean)")
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(max_sharpe_stats[0][1], max_sharpe_stats[0][0], marker="*", markersize=30, markeredgecolor="black",
             markerfacecolor="gold")
    plt.plot(min_risk_stats[0][1], min_risk_stats[0][0], marker="*", markersize=30, markeredgecolor="black",
             markerfacecolor="violet")
    plt.title(f"Scatter plot of {NUM_PORTFOLIOS} randomly generated portfolios with varying ratio of assets")
    st.pyplot()
    st.write("+ Yellow star: Estimate of optimal portfolio with minimum risk")
    st.write("+ Purple star: Estimate of optimal portfolio with highest return for a given level of risk")
    st.write("***")


def plot_pie_chart_max_sharpe(optimal_ratio_max_sharpe_, stock_list_):

    # sort the ticker array and ratio array together in ascending order
    zipped_arrays = zip(optimal_ratio_max_sharpe_, stock_list_)
    sorted_pairs_ascending = sorted(zipped_arrays)

    # create reverse of array:
    sorted_pairs_ascending_reverse = sorted_pairs_ascending[::-1]

    # small ratio between large ratios
    sorted_pairs_between = []
    count = 0
    while count < np.floor(len(sorted_pairs_ascending)):
        sorted_pairs_between.append(sorted_pairs_ascending[count])
        sorted_pairs_between.append(sorted_pairs_ascending_reverse[count])
        count += 1

    # the above while loop creates a new array twice the elements of the input array,
    # need to return original length of the array
    sorted_pairs_between = sorted_pairs_between[:len(sorted_pairs_ascending)]

    # extract the sorted arrays from tuples
    tuples = zip(*sorted_pairs_between)
    optimal_ratio_max_sharpe_, stock_list_ = [list(t) for t in tuples]
    print(optimal_ratio_max_sharpe_)
    print(stock_list_)

    # pie chart of the portfolio with highest return for a given level of risk
    col = ['#b7ffdf', '#3adaa2', '#6cd7dc', '#3d85c6', '#b0afe6', '#d8c8e9', '#ffdbe4', '#fcbbbb', '#faa17a',
           '#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#ffb3ba']
    explode_array = []
    for index in range(len(STOCK_LIST)):
        explode_array.append(0.02)
    patches, texts, autotexts = plt.pie(optimal_ratio_max_sharpe_, labels=stock_list_, normalize=True,
                                        radius=0.6, explode=explode_array, autopct="%0.2f%%", pctdistance=0.8,
                                        wedgeprops={'linewidth': 0, 'edgecolor': 'black'}, colors=col)
    unused_variable = [text.set_color('black') for text in texts]
    unused_variable = [autotext.set_color('black') for autotext in autotexts]
    # plt.title("Maximum Sharpe Ratio Portfolio")
    plt.axis("equal")

    # plot pie chart w header
    st.subheader('Portfolio with Maximum Sharpe Ratio')
    col3, col4 = st.columns((6, 1))
    col3.pyplot()

    # display ratio table
    tuples1 = zip(*sorted_pairs_ascending_reverse)
    optimal_ratio_max_sharpe_display, stock_list_display = [list(f) for f in tuples1]
    optimal_ratio_max_sharpe_display = [("%.2f" % (item * 100)) + "%" for item in optimal_ratio_max_sharpe_display]
    display_max_sharpe_df = pd.DataFrame({'Asset': stock_list_display, "Ratio": optimal_ratio_max_sharpe_display})
    display_max_sharpe_df.index = range(1, len(display_max_sharpe_df) + 1)
    col4.table(display_max_sharpe_df)

    # display portfolio stats
    col5, col6, col7 = st.columns(3)
    col5.metric("Annual Expected Return (Mean)", f"{round(max_sharpe_stats.reshape(-1)[0] * 100, 3)} %")
    col6.metric("Annual Volatility (Std Dev)", f"{round(max_sharpe_stats.reshape(-1)[1] * 100, 3)} %")
    col7.metric("Sharpe Ratio", round(max_sharpe_stats.reshape(-1)[2], 2))

    st.write("***")


def plot_pie_chart_min_risk(optimal_ratio_min_risk_, stock_list_):
    # sort the ticker array and ratio array together in ascending order
    zipped_arrays = zip(optimal_ratio_min_risk_, stock_list_)
    sorted_pairs_ascending = sorted(zipped_arrays)

    # create reverse of array:
    sorted_pairs_ascending_reverse = sorted_pairs_ascending[::-1]

    # small ratio between large ratios
    sorted_pairs_between = []
    count = 0
    while count < np.floor(len(sorted_pairs_ascending)):
        sorted_pairs_between.append(sorted_pairs_ascending[count])
        sorted_pairs_between.append(sorted_pairs_ascending_reverse[count])
        count += 1

    # the above while loop creates a new array twice the elements of the input array,
    # need to return original length of the array
    sorted_pairs_between = sorted_pairs_between[:len(sorted_pairs_ascending)]

    # extract the sorted arrays from tuples
    tuples = zip(*sorted_pairs_between)
    optimal_ratio_min_risk_, stock_list_ = [list(t) for t in tuples]
    print(optimal_ratio_min_risk_)
    print(stock_list_)

    # pie chart of the portfolio with lowest risk
    col = ['#b7ffdf', '#3adaa2', '#6cd7dc', '#3d85c6', '#b0afe6', '#d8c8e9', '#ffdbe4', '#fcbbbb', '#faa17a',
           '#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#ffb3ba']
    explode_array = []
    for index in range(len(STOCK_LIST)):
        explode_array.append(0.02)
    patches, texts, autotexts = plt.pie(optimal_ratio_min_risk_, labels=stock_list_, normalize=True,
                                        radius=0.5, explode=explode_array, autopct="%0.2f%%", pctdistance=0.8,
                                        wedgeprops={'linewidth': 0.2, 'edgecolor': 'black'}, colors=col)
    unused_variable = [text.set_color('black') for text in texts]
    unused_variable = [autotext.set_color('black') for autotext in autotexts]
    # plt.title("Highest Return for Minimum Risk Portfolio")
    plt.axis("equal")

    # plot pie chart w header
    st.subheader('Portfolio with Minimum Risk')
    col8, col9 = st.columns((6, 1))
    col8.pyplot()

    # display ratio table
    tuples1 = zip(*sorted_pairs_ascending_reverse)
    optimal_ratio_min_risk_display, stock_list_display = [list(f) for f in tuples1]
    optimal_ratio_min_risk_display = [("%.2f" % (items * 100)) + "%" for items in optimal_ratio_min_risk_display]
    display_min_risk_df = pd.DataFrame({'Asset': stock_list_display, "Ratio": optimal_ratio_min_risk_display})
    display_min_risk_df.index = range(1, len(display_min_risk_df) + 1)
    col9.table(display_min_risk_df)

    # display portfolio stats
    col10, col11, col12 = st.columns(3)
    col10.metric("Annual Expected Return (Mean)", f"{round(min_risk_stats.reshape(-1)[0] * 100, 3)} %")
    col11.metric("Annual Volatility (Std Dev)", f"{round(min_risk_stats.reshape(-1)[1] * 100, 3)} %")
    col12.metric("Sharpe Ratio", round(min_risk_stats.reshape(-1)[2], 2))

    st.write("***")


# COMPOUNDING RETURNS / EQUITY CURVE PLOTTING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def download_data(start_date, end_date, stock_list):
    # get data from yahoo finance
    stock_data_ = {}

    for stock in stock_list:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data_[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data_).interpolate()


def calculate_return(data):
    # get % change daily returns
    daily_return = data / data.shift(1)
    return daily_return[1:].to_numpy()


def weighted_capital(weights):
    # divide starting capital by portfolio weights
    weighted_starting_capital = []
    for w in weights:
        weighted_starting_capital.append(START_CAPITAL * w)
    return np.array(weighted_starting_capital)


def portfolio_df(wsc, dr, stock_data, stock_list):
    # create an array containing the price of assets owned, compounded over time:
    array = []

    g = 0
    while g < len(wsc):
        i = 0
        x = wsc[g]
        while i < len(dr):
            x = x * dr[i][g]
            array.append(x)
            i += 1
        g += 1

    array = np.array(array)

    # shape the array into 2D array
    array = array.reshape(len(wsc), len(dr))

    # make the array into a dataframe, transpose it such that the cols are prices of individual assets over time:
    df = pd.DataFrame(array).transpose()

    # insert weighted starting capital at first row of dataframe:
    df = pd.concat([pd.DataFrame(wsc).transpose(), df])

    # insert new column which is the sum of all prices horizontally in a single row (axis = 1):
    df["Portfolio Value"] = df.iloc[:, :].sum(axis=1)

    # taking the date index from stock_data, replace row index of new dataframe:
    df["Date"] = pd.DataFrame(stock_data.index)
    df.set_index("Date", inplace=True)
    print(df)

    # create dataframe with only the portfolio data:
    df = df.drop([i for i in range(len(stock_list))], axis=1)
    return df


def benchmark_portfolio(start_date, end_date):
    # get data from yahoo finance and store it in a pandas dataframe
    compared_data = yf.download(compared_stock, start_date, end_date)['Close']
    compared_data = compared_data.interpolate()

    # get % change of daily returns
    compared_data_return = compared_data / compared_data.shift(1)
    compared_data_return = compared_data_return[1:]

    print(compared_data_return)
    print(type(compared_data_return))

    capital = START_CAPITAL
    capital_array = [START_CAPITAL]
    for i in compared_data_return[:]:
        capital = capital * i
        capital_array.append(capital)

    print(pd.DataFrame(capital_array))
    capital_dataframe = pd.DataFrame(capital_array)

    # taking the date index from compared_data, replace row index of capital_dataframe:
    capital_dataframe["Date"] = pd.DataFrame(compared_data.index)
    capital_dataframe.set_index("Date", inplace=True)
    print(capital_dataframe)

    capital_dataframe.columns = [compared_stock]
    print(capital_dataframe)

    return capital_dataframe


def plot_equity_curve(start_date, end_date, stock_list, optimal_ratio_max_sharpe, optimal_ratio_min_risk):

    stock_list = stock_list
    w_max_sharpe = optimal_ratio_max_sharpe
    w_min_risk = optimal_ratio_min_risk
    start_date = start_date
    end_date = end_date

    stock_data = download_data(start_date, end_date, stock_list)
    print(stock_data)

    daily_returns = calculate_return(stock_data)
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    print(daily_returns)
    daily_returns[np.isnan(daily_returns)] = 1
    print(daily_returns)

    weights = w_max_sharpe
    weighted_start_capital_max_sharpe = weighted_capital(weights)
    weights = w_min_risk
    weighted_start_capital_min_risk = weighted_capital(weights)
    print(weighted_start_capital_max_sharpe)
    print(weighted_start_capital_min_risk)

    weighted_start_capital = weighted_start_capital_max_sharpe
    portfolio_df_max_sharpe = portfolio_df(weighted_start_capital, daily_returns, stock_data, stock_list)
    print(portfolio_df_max_sharpe)

    weighted_start_capital = weighted_start_capital_min_risk
    portfolio_df_min_risk = portfolio_df(weighted_start_capital, daily_returns, stock_data, stock_list)
    print(portfolio_df_min_risk)

    portfolio_df_benchmark = benchmark_portfolio(start_date, end_date)

    a, = plt.plot(portfolio_df_benchmark, label='SNP500', color='cyan', linewidth=1)
    b, = plt.plot(portfolio_df_max_sharpe, label='Optimal portfolio with max sharpe ratio', color='gold', linewidth=1)
    c, = plt.plot(portfolio_df_min_risk, label='Optimal portfolio with min risk', color='violet', linewidth=1)
    plt.title(f"Equity curve of optimal portfolios vs SnP500 index (Starting capital = ${START_CAPITAL})")
    plt.legend(handles=[a, b, c])
    plt.xlabel("Date")
    plt.ylabel("Equity")
    st.subheader("Equity curve of optimal and minimum risk portfolio, benchmarked to 100 % Snp 500 portfolio")
    st.write("")
    st.pyplot()

    a, = plt.plot(portfolio_df_benchmark, label='SNP500', color='cyan', linewidth=1)
    b, = plt.plot(portfolio_df_max_sharpe, label='Optimal portfolio with max sharpe ratio', color='gold', linewidth=1)
    c, = plt.plot(portfolio_df_min_risk, label='Optimal portfolio with min risk', color='violet', linewidth=1)
    plt.title(f"Equity curve of optimal portfolios vs SnP500 index (Starting capital = ${START_CAPITAL})")
    plt.legend(handles=[a, b, c])
    plt.xlabel("Date")
    plt.ylabel("Equity (Logarithmic Scale)")
    plt.yscale('log', base=10)
    with st.expander("Display logged graph"):
        st.pyplot()
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # UI stuff, user input global variables:
    with st.form(key='form1'):
        sel = []
        st.subheader("1. Select Stocks / ETF(gold, bonds, stocks) / Cryptocurrencies:")
        stock_input = st.multiselect("Select Stocks (NYSE/NASDAQ/AMEX)", df_excel_stock["Ticker + Name"], key=1,
                                     default=default_list1)
        etf_input = st.multiselect("Select ETFs (NYSE/NASDAQ/AMEX)", df_excel_etf["Ticker + Name"], key=2,
                                   default=default_list2)
        crypto_input = st.multiselect("Select Cryptocurrencies (Top 100 only)", df_excel_crypto["Ticker + Name"], key=3,
                                      default=default_list3)

        def append_ticker(x):
            # maybe there is a better way to avoid error that happens when multiselect widget is left empty?
            try:
                for j in pd.DataFrame(x)[0]:
                    sel.append(j)
            except KeyError:
                pass

        append_ticker(stock_input)
        append_ticker(etf_input)
        append_ticker(crypto_input)

        st.write("***")
        st.subheader("2. Select date range")
        c1, c2 = st.columns(2)
        with c1:
            START_DATE = st.date_input("Start date", dt.date(2000, 1, 1))
        with c2:
            END_DATE = st.date_input("End date", dt.date.today(), max_value=dt.date.today())

        st.write("***")
        st.subheader("3. Select number of randomly generated portfolios:")
        NUM_PORTFOLIOS = st.slider(" ", 100000, 500000, 10)
        with st.expander("Tip:"):
            st.write("~ 50000 for faster results, increase slider to improve estimate of optimal asset weights")

        submitted = st.form_submit_button(label='Enter')

    if submitted:
        ticker_index = []

        for i in sel:
            a = df_concat.loc[df_concat["Ticker + Name"] == str(i)]
            ticker_index.append(str(a.index[0]))
        for i in ticker_index:
            STOCK_LIST.append(df_concat.iloc[int(i), 0])

        if len(STOCK_LIST) <= 1:
            st.error("Please enter 2 or more assets!")
            st.stop()
        else:
            pass
        # debug:
        # st.write(STOCK_LIST)
    else:
        st.stop()

    dataset = get_dataset()

    newest_ticker = set_new_date_for_available_data()

    dataset_correct_date = get_dataset()

    dataset_final = dataset_correct_date.backfill()

    plot_dataset(dataset_final)

    log_daily_return = get_normalised_daily_return(dataset_final)

    # debug
    # st.write(dataset)

    mean_return_annual, cov_matrix_annual = annualised_mean_covariance(log_daily_return)

    display_mean_covariance_table()

    st.spinner(text='In progress...')

    correlation_heatmap(log_daily_return)

    p_weights, p_means, p_stdDevs = generate_random_portfolios()

    optimal_ratio_max_sharpe, max_sharpe_stats, optimal_ratio_min_risk, min_risk_stats\
        = calculate_portfolio_max_sharpe_min_risk(p_means, p_stdDevs)

    # debug
    # st.write("")
    # st.write(f"Asset ratio that yields highest sharpe ratio is {optimal_ratio_max_sharpe}")
    # st.write(f"Its expected return (mean), volatility (SD) and sharpe ratio are {max_sharpe_stats.reshape(-1)}")
    # st.write("")
    # st.write(f"Asset ratio that yields lowest risk is {optimal_ratio_min_risk}")
    # st.write(f"Its expected return (mean), volatility (SD) and sharpe ratio are {min_risk_stats.reshape(-1)}")
    # st.write("")

    scatter_plot_optimal_portfolios(p_means, p_stdDevs)

    plot_pie_chart_max_sharpe(optimal_ratio_max_sharpe, STOCK_LIST)

    plot_pie_chart_min_risk(optimal_ratio_min_risk, STOCK_LIST)

    plot_equity_curve(START_DATE, END_DATE, STOCK_LIST, optimal_ratio_max_sharpe, optimal_ratio_min_risk)