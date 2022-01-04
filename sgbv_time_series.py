import pandas as pd
import numpy as np
import streamlit as st
from streamlit_disqus import st_disqus
import matplotlib.pyplot as plt, mpld3
import altair as alt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from causalimpact import CausalImpact
from openpyxl import load_workbook
import datetime
from datetime import datetime, date, time
from dateutil.relativedelta import relativedelta
import calendar

import os
import os.path
from io import StringIO, BytesIO
import base64
from PIL import Image

import ruptures as rpt #for changepoint detection
import pingouin as pt # for two-sample paired t-test ITSA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import acf, acovf, pacf, pacf_ols, pacf_yw, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel

import pmdarima as pm
from pmdarima import auto_arima

st.markdown("# Application Title TBD")

about_tool = """
   This tool is intended for quick and ready time series analyses.
   It has been developed primary for use by sexual violence researchers who do not routinely
   use time series methods. The time series approaches in this tool include interrupted time series,
   seasonal autoregressive integrated moving averages (SARIMAX), Bayesian structural time series (BSTS).

   Before you start, ensure that your dataset has at least two columns. One of the columns
   must be a date column. One of the other columns should be the primary
   outcome that you are interested in analyzing e.g., domestic violence cases. The data in this
   column should be in numeric form -- either float or integer. At a minimum, the tool will give univariate
   time series results.

   You must select start, intervention, and end dates on the side panel before you start.
   There is an option to draw a map for the reporting facilities/units.
   The tool is coded in Python v3.7.0. It would be great to cite Python in your work.

   The technical notes are included for your convenience in interpreting the findings.
    """
st.markdown("#### Read the following section before using the tool")
start = st.expander("ReadMe")
start.write(about_tool)

st.markdown("-----")

st.sidebar.markdown("## Menu Panel")
#st.sidebar.markdown("Select dates for analysis")
start_date = st.sidebar.date_input("Choose start date for your series: ")
shock_date = st.sidebar.date_input("Choose lockdown start date: ")
end_date = st.sidebar.date_input("Choose end date for your series: ")
num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
pre_months = (shock_date.year - start_date.year) * 12 + (shock_date.month - start_date.month) + 1
post_months = (end_date.year - shock_date.year) * 12 + (end_date.month - shock_date.month)
notation = f"""Your dataset has {num_months} observation months, with a pre-period of {pre_months} and a
post-period of {post_months} months"""
st.sidebar.markdown(notation)


uploaded_file = st.file_uploader("Upload main dataset", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    #df = pd.read_csv(uploaded_file, parse_dates=["date2"])
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.rename(columns={col: "date2"})
            except ValueError:
                pass

    df = df.set_index("date2")
    df = df.loc[:, (df !=0).any(axis=0)]
    df.index.freq = "MS"
    variables = list(df)
    outcome = st.sidebar.selectbox("Choose outcome variable", variables)

if st.checkbox("Show Uploaded Data", False):
    st.write(df)

coordinates = st.file_uploader("Optional: upload facility coordinates file", type=["csv", "xls", "xlsx"])

if coordinates is not None:
    coords = pd.read_csv(coordinates)

if st.checkbox("Show Coordinates Data", False):
    st.write(coords)

# Check if dates make sense

def date_checker(start_date, end_date, shock_date):
    assert(shock_date > start_date), "Intervention date cannot be earlier than or equal to the start date"
    assert(end_date > shock_date), "End date cannot be earlier than or equal to the intervention date"

st.markdown("-----")
st.markdown('## Explore Data and Conduct Analyses')
st.sidebar.subheader('Exploratory Data Analysis')
st.markdown("*Check the boxes on the side panel to explore and conduct the desired analyses.*")

if st.sidebar.checkbox('Data summary'):
    date_checker(start_date, end_date, shock_date)
    df = df.loc[start_date : ]
    if st.sidebar.checkbox("Summary"):
        line = alt.Chart(df.reset_index()).mark_line().encode(
                 x=alt.X("date2", axis=alt.Axis(title="")),
                 y=outcome,
                 tooltip=[outcome, "date2"]
             ).interactive()
        rule = alt.Chart(pd.DataFrame({
                "Date": [shock_date],
                "color": ["red"]})).mark_rule(strokeDash=[2,2]).encode(
                x = alt.X("Date:T", axis=alt.Axis(title="")),
                color=alt.Color('color:N', scale=None)
                )
        graph = line + rule
        col1, col2 = st.columns(2)
        col1.caption("Summary statistics")
        col1.write(df[outcome].describe().apply('{:.2f}'.format))

        col2.caption(f"{outcome} trends")
        col2.write(graph)

        technical_note = """
        Visualize the trends to assess for outliers, seasonality.
        Check distributions. If the data are highly skewed, may need to normalize.
        The program does this automatically for you.
        This is useful for appropriate model selection and for additional data preprocessing.
        For higher frequency seasonality i.e., less than a month, don't use SARIMAX
        """
        expander_note = st.expander("Technical notes")
        expander_note.write(technical_note)

    if st.sidebar.checkbox('Missing value(s) Check'):
        st.subheader('Missing values')
        st.write(df[outcome].isnull().sum())

    if st.sidebar.checkbox("Exploratory plots"):
        df["year"] = pd.DatetimeIndex(df.index).year.astype(int)
        df["month"] = pd.DatetimeIndex(df.index).month.astype(int)
#         fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi=150)
#         sns.boxplot(x="month", y=outcome, data=df)
#         sns.boxplot(x="year", y=outcome, data=df, ax=axes[0])
#         plt.suptitle(f"Trends and monthly variation in {outcome}", fontsize=24)
#         st.write(fig)
        note = f"""
        The monthly boxplot indicates monthly variation in {outcome} and may indicate
        seasonality. The annual boxplot indicates changes in the long term trends in
        {outcome}. These boxplots also help identify outliers. Seasonality and trends are futher
        explored under the time series decomposition in the technical section. Seasonality and trends
        need to be adjusted for to elicit the impact of the policy of interest.
        The kernel density plots show distribution of the variable. It is not strictly important for the
        variable to be normally distributed before analyses although some authors insist on this.
        """
        expander_boxplot = st.expander("Technical notes: boxplots")
        expander_boxplot.write(note)

        monthly_boxplot = alt.Chart(df).mark_boxplot().encode(
            x="month:O",
            y=alt.Y(f"{outcome}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("month:O", legend=None))
        annual_boxplot = alt.Chart(df).mark_boxplot().encode(
            x="year:O",
            y=alt.Y(f"{outcome}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("year:O", legend=None))

        col1, col2 = st.columns(2)
        expdr1 = col1.expander(f"Monthly variation {outcome} boxplot")
        expdr2 = col2.expander(f"Annual variation {outcome} boxplot")
        expdr1.write(monthly_boxplot)
        expdr2.write(annual_boxplot)

        df1 = df.pivot(columns="month", values=outcome).apply(lambda x: x.dropna().reset_index(drop = True))
        df2 = df.pivot(columns="year", values=outcome).apply(lambda x: x.dropna().reset_index(drop = True))

        fig, axes = plt.subplots(1,2, figsize=(20,7), dpi=150)
        for x in range(df.month.min(), df.month.max() + 1):
            sns.kdeplot(data=df1[x], shade=True, ax=axes[0])

        for x in range(df.year.min(), df.year.max() + 1):
            sns.kdeplot(data=df2[x], shade=True, ax=axes[1])

        plt.suptitle(f"Monthly and annual kdensity plots {outcome}")
        expander_kde = st.expander(f"Kdensity plots {outcome}")
        expander_kde.write(fig)


    if st.sidebar.checkbox("Show Facility Map", False):
        st.map(coords)
st.sidebar.subheader('Technical Checks')
if st.sidebar.checkbox("Technical checks"):

    date_checker(start_date, end_date, shock_date)
    c_point = df.loc[start_date:] # Dataset for changepoint detection
    df = df.loc[start_date : shock_date]
    # test-train split -- use 70:30 split
    split = round(pre_months * .7)
    train = df[:split]
    test = df[-split:]

    technical_note = """
        - Time series data in public health typically have three features: long term trends, seasonal variations e.g.,
          in malaria or respiratory tract cases, and random noise.
        - Box and Cox recommended a minimum of 50 pre-period observations for time series analyses. Other authors recommend
          between 60 and 110 observations. The math behind these recommendations is unclear, but simulations show that
          estimates become more stable with increasing observations.
        - The key idea in assessing the effect of a shock (policy, intervention, crisis) is to estimate the counterfactual
          (what would have happened in the absence of the shock) and separate that from what is observed
        - The analyst also needs to think about other competing issues like data measurement changes, health worker strikes,
          contemporary government policies, and account for them. In this tool, we use Chow tests to assess for these other issues
          that we refer to us structural breaks.
        - Click on the images below to have a sense of the key issues in time series analyses. These are stylized examples. The
          trend can be increasing, decreasing or flat. The policy change can be immediate, slow, immediate with reversal to mean, or other variations
        - Performs test:train splits, evaluates model performance using RMSE and AIC, and checks for residuals
        - Test:train split done on the pre-shock data as 70:30
        - Check for stationarity, autocorrelations
        - The decomposition process uses locally estimated scatterplot smoothing to identify trends, seasonality, and residuals
        - Check if the decomposed trend is increasing, decreasing and the rate of change
        - Check for seasonality i.e., repetitive patterns at similar months over years
        - If the seasonal pattern is growing with time -- you'll need multiplicative
        decomposition. The tool will do this for you.
        - You should have no discernible patterns in the residual plot
        - This section also contains a changepoint (structural break) detection algorithm

    """
    expander_note = st.expander("Technical notes")
    expander_note.write(technical_note)

    col1, col2 = st.columns(2)
    img1 = Image.open("/Users/wobiero/Desktop/Sex violence/time_series_technical.jpg")
    img2 = Image.open("/Users/wobiero/Desktop/Sex violence/technical_issues_2.jpg")
    col1.image(img1, use_column_width=True)
    col2.image(img2, use_column_width=True)

    if st.checkbox('Time series technical issues'):
        st.subheader('Time series explanatory technical graph')
        img1 = Image.open("/Users/wobiero/Desktop/Sex violence/time_series_technical.jpg")
        st.image(img1, caption="Time series technical issues")

    df["6-month-MA"]= df[outcome].rolling(window=6).mean()
    df["12-month-MA"]= df[outcome].rolling(window=12).mean()
    df["18-month-MA"]= df[outcome].rolling(window=18).mean()
    df["ewm"] = df[outcome].ewm(span=12, adjust=False).mean()
    df["date"] = df.index

    fig2 = alt.Chart(df).mark_line().transform_fold(
        fold=['ewm',"12-month-MA", "6-month-MA", 'Sexual Violence'],
        as_=['variable', 'value']
    ).encode(
        x='yearmonth(date):T',
        y='max(value):Q',
        color='variable:N'
    ).interactive()
    expander_moving_averages = st.expander(f"Moving Averages Plot: {outcome}")
    expander_moving_averages.write(fig2)

    # Check number of lags using KPSS test
    def kpss_test(series, **kw):
        statistic, p_value, n_lags, critical_values = kpss(series, **kw)
        return statistic, p_value
    kpss_values = [kpss_test(diff(df[outcome], k_diff=i).dropna()) for i in range(24)]
    p_values = [x[1] for x in kpss_values]
    kpss_stats = [x[0] for x in kpss_values]
    lag = [n for n, i in enumerate(p_values) if i >.05][1]


    kpss_statement = f'The KPSS statistic for a stationary series is {kpss_stats[lag]:.3f} with a p-value of {p_values[lag]} and the number of required lags are {lag}'
    expander_kpss = st.expander("KPSS Statistics")
    expander_kpss.write(kpss_statement)

    if st.checkbox("Autocorrelation function plots", False):

        col1, col2 = st.columns(2)
        ax1 = plot_acf(diff(df[outcome], k_diff=0).dropna())
        ax2 = plot_acf(diff(df[outcome], k_diff=24).dropna())
        col1.caption(f"Undifferenced ACF plot {outcome}")
        col2.caption(f"Differenced ACF Plot {outcome} with {lag} lags")
        col1.write(ax1)
        col2.write(ax2)

        col3, col4 = st.columns(2)
        ax1 = plot_pacf(diff(df[outcome], k_diff=0).dropna())
        ax2 = plot_pacf(diff(df[outcome], k_diff=24).dropna())
        col3.caption(f"Undifferenced PACF plot {outcome}")
        col4.caption(f"Differenced PACF Plot {outcome} with {lag} lags")
        col3.write(ax1)
        col4.write(ax2)

    try:
        decomposition = seasonal_decompose(df[outcome], model="additive", period=12) #Automate number of lags from ARIMA function
    except AttributeError:
        error = """
                    Your Date column is not in proper format
                    """
        raise AttributeError(error)

    fig, ax = plt.subplots(figsize=(8,3.5), dpi=350)
    decomposition.trend.plot(c="b")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.title(f"Decomposed trend -- {outcome}")
    plt.xlabel("")
    plt.ylabel("Monthly cases")
    plt.axvline(x=shock_date, ls="--", color="r")
    plt.grid(axis="both", lw=.5, alpha=.2)

    expander_trend = st.expander("Series decomposition: trend plot")
    expander_trend.write(fig)


    fig, ax = plt.subplots(figsize=(8,3.5), dpi=350)

    decomposition.seasonal.plot(c="b")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.title(f"Seasonality Plot -- {outcome}")
    plt.xlabel("")
    plt.ylabel("Monthly variation")
    plt.axhline(0, ls ="--", color="r")
    plt.axvline(x=shock_date, ls="--", color="r")
    plt.grid(axis="both", lw=.5, alpha=.2)
    expander_seasonality = st.expander("Series decomposition: seasonality plot")
    expander_seasonality.write(fig)

    seasonal_variation = decomposition.seasonal[:12].to_frame()
    seasonal_variation["month"] = pd.DatetimeIndex(seasonal_variation.index).month.astype(int)
    seasonal_variation["month"] = seasonal_variation["month"].apply(lambda x: calendar.month_abbr[x])
    seasonal_variation = seasonal_variation.set_index("month")
    seasonal_variation = seasonal_variation.T
    expander_seas_var = st.expander(f"Seasonal variations results: {outcome}")
    expander_seas_var.dataframe(seasonal_variation)


    fig, ax = plt.subplots(figsize=(8,3.5), dpi=350)
    decomposition.resid.plot(c="b")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.title(f"Residual Plot -- {outcome}")
    plt.xlabel("")
    plt.ylabel("Cases")
    plt.axvline(x=shock_date, ls="--", color="r")
    plt.axhline(0, ls ="--", color="r")
    plt.grid(axis="both", lw= .5, alpha=.2)

    expander_residual = st.expander("Series decomposition: residual plot")
    expander_residual.write(fig)

    # Changepoints detection using Binary Segmentation Search
    data_points = np.array(c_point[outcome])
    model = "l2"
    algo = rpt.Binseg(model=model).fit(data_points)
    my_bkps = algo.predict(n_bkps=6)
    c_point.reset_index(inplace=True)
#     fig, ax = plt.subplots(figsize=(12,5))
#     plt.plot(c_point.index, c_point[outcome], label="Observed")
#     plt.axvline(my_bkps[0], ls="--", color="k")
#     plt.axvline(my_bkps[1], ls="--", color="k")
#     plt.axvline(my_bkps[2], ls="--", color="k")
#     plt.axvline(my_bkps[3], ls="--", color="k")
#     plt.axvline(my_bkps[4], ls="--", color="k")
#     plt.axvline(my_bkps[5], ls="--", color="k", label="Breakpoint")
#     plt.legend()
#     plt.xlabel("""Months
#     Note: The breakpoints are detected using a binary segmentation search algorithm""")
#     plt.title(f"Structural breaks: {outcome}")

    c_point["date2"] = pd.to_datetime(c_point["date2"])
    my_bkps = [x - 1 for x in my_bkps]
    breakdates = c_point["date2"].iloc[my_bkps]
    breakdates = list(breakdates)

    line = alt.Chart(c_point.reset_index()).mark_line().encode(
                 x=alt.X("date2", axis=alt.Axis(title="")),
                 y=outcome,
                 tooltip=[outcome, "date2"]
             ).interactive()
    rules = alt.Chart(pd.DataFrame({
        "Date": breakdates,
    })).mark_rule().encode(
        x="Date:T",
        strokeDash="Date",
        color=alt.value("red")
    )

    fig = line + rules
    st_breaks = st.expander("Structural Breaks Graph")
    st_breaks.write(fig)
    change_point_note = r"""
    ###### The structural breaks above are identified using a binary segmentation search
    algorithm. These breaks often point to competing factors that could drive the results
    and that may not be readily apparent. That is, these breaks can shift the overall
    trends for the data and lead to biased estimates of the impact of the shock being assessed.
    The technical analysis for these breaks should be guided
    by visualization of basic trends and a contextual understanding of the data.
    See page 24 of [Truong et al, (2020)](http://www.laurentoudre.fr/publis/TOG-SP-19.pdf)
    for technical details around binary segmentation searches and its alternatives.
    The implementation in this tool is done using the Ruptures library in Python.
    """
    cp_note = st.expander("Structural breaks notes")
    cp_note.write(change_point_note)

st.sidebar.subheader('Conduct analyses')
if st.sidebar.checkbox('Time series analyses'):
    date_checker(start_date, end_date, shock_date)
    df = df.loc[start_date:]
    menu = ["Seasonal Holt-Winters",
            "Bayesian Structural Time Series",
            "Interrupted Time Series",
            "SARIMAX"]
    menu_choices = st.sidebar.selectbox("Select time series method", menu)
    if menu_choices == "Seasonal Holt-Winters":
    #if st.sidebar.checkbox('Seasonal Holt-Winters'):
        st.subheader("Seasonal Holt-Winters")
        note = """
        This function uses the Statsmodels library to conduct triple exponential SHW
        """
        alpha = 2/(pre_months + 1)
        df2 = df.loc[start_date : shock_date]
        shw_model = ExponentialSmoothing(df2[outcome],
                                        trend="mul", seasonal="mul",
                                       seasonal_periods=12).fit()
        predictions = shw_model.forecast(post_months)

#         fig, ax = plt.subplots(figsize=(12,5), dpi=150)
#         plt.plot(df.index[:pre_months], df[outcome][:pre_months])
#         plt.plot(df.index[-pre_months:], df[outcome][-pre_months:])
#         predictions.plot(label="SHW prediction")
#         plt.legend()
#         st.write(fig)

        observed = df[outcome][-post_months:]
        shw_results = pd.concat([observed, predictions], axis=1)
        shw_results.columns = ["observed", "shw_predictions"]
        shw_results["date"] = shw_results.index
        shw_results["change"] = shw_results["observed"] - shw_results["shw_predictions"]
        if st.checkbox("Show Seasonal Holt Winters Results", False):
            st.write(shw_results)


        fig2 = alt.Chart(shw_results).mark_line().transform_fold(
            fold=["observed", "shw_predictions"],
            as_=['variable', 'value']
        ).encode(
            x='yearmonth(date):T',
            y='max(value):Q',
            color='variable:N'
        ).interactive()
        expander_shw_graph = st.expander(f"Seasonal Holt-Winters Plot: {outcome}")
        expander_shw_graph.write(fig2)

        cumulative_observed = shw_results["observed"].sum().round()
        cumulative_predicted = shw_results["shw_predictions"].sum()
        cumulative_change = shw_results["change"].sum().round()
        monthly_mean_change = shw_results["change"].mean()
        t_stat, p_value = ttest_rel(shw_results["observed"], shw_results["shw_predictions"])
        monthly_std_error = abs(monthly_mean_change/t_stat)
        monthly_upper = monthly_mean_change + monthly_std_error
        monthly_lower = monthly_mean_change - monthly_std_error
        monthly_pred_mean = shw_results["shw_predictions"].mean()
        a = "statistically significant"
        b = "statistically insignificant"
        p = .05

        shw_narrative = f"""
        There were {cumulative_observed:,.1f} observed {outcome} cases during the post (intervention) period.
        In the counterfactual scenario using a seasonal Holt-Winters approach, {cumulative_predicted:,.1f} cases
        were expected during the post period, with a cumulative change of {cumulative_change:,.1f} cases.

        This is equivalent to a {a if p_value < p else b} mean monthly change of
        {monthly_mean_change:,.1f} [{monthly_upper:,.1f}, {monthly_lower:,.1f}] cases,
        with a p-value of {p_value:,.3f}. This represents a relative change of
        {monthly_mean_change/monthly_pred_mean * 100:.1f}% [{monthly_lower/monthly_pred_mean * 100:.1f}%,
        {monthly_upper/monthly_pred_mean * 100:.1f}%] over the expected baseline.

        The interpretation of the impact results should be based on
        a good understanding of the study context including how the data were collected, changes in reporting
        standards, government policy changes and competing events like population displacements and health worker
        strikes that could influence the results.
        """
        expander_shw_results = st.expander("Seasonal Holt-Winters Summary")
        expander_shw_results.write(shw_narrative)

        new_df = shw_results.to_csv().encode('utf-8')
        st.download_button(
            "Download seasonal Holt-Winters estimates",
            new_df,
            "shw.csv",
            key="download-csv"
        )

    elif menu_choices == "Bayesian Structural Time Series":
    #if st.sidebar.checkbox('Bayesian Structural Time Series'):
        st.subheader('Bayesian Structural Time Series')
        """
        This function uses the tensorflow causal impact library by
        Willian Fuks to construct Bayesian structural time series models.
        Kindly cite his work.
        """
        pre_period = [0, pre_months-1]
        post_period = [pre_months, num_months-1]
        shock = pre_months
        df.reset_index(inplace=True)
        df = df[[outcome]]
        ci = CausalImpact(df.astype(int).astype(float), pre_period, post_period)
        expander_bsts_0 = st.expander("BSTS Regression Summary")
        expander_bsts_0.write(ci.summary())
        expander_bsts_1 = st.expander("BSTS Regression Summary Report")
        expander_bsts_1.write(ci.summary(output="report"))
        expander_bsts_2 = st.expander("BSTS Results Detailed")
        expander_bsts_2.write(ci.inferences)

        fig, ax = plt.subplots(figsize=(10,5), dpi=200)
        plt.plot(ci.inferences.index[1:], ci.inferences["point_effects_means"][1:], c="k", label="Point Effects")
        plt.fill_between(ci.inferences.index[1:], ci.inferences["point_effects_upper"][1:],
                         ci.inferences["point_effects_lower"][1:], color="green", alpha=.3)
        #plt.vlines(shock, minimum, maximum, ls="--", colors="r")
        plt.axvline(x=shock, ls="--", color="r")
        plt.axhline(y=0, color="r", ls="--")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.grid(axis="both", lw=.5, alpha=.2)
        plt.legend(loc="upper left", fancybox=True)
        plt.xlabel("Time in Months")
        plt.ylabel("Monthly case change against baseline")
        expander_bsts_3 = st.expander("BSTS Point effects")
        expander_bsts_3.write(fig)

        ci_data = ci.inferences.fillna(0)

        fig, ax = plt.subplots(figsize=(10,5), dpi=200)
        plt.plot(ci_data.index, ci_data["post_cum_effects_means"], ls="--", color="k", label="Cumulative Effects")
        plt.fill_between(ci_data.index[1:], ci_data["post_cum_effects_upper"][1:],
                         ci_data["post_cum_effects_lower"][1:], color="green", alpha=.3)
        plt.axvline(x=shock, ls="--", color="r")
        plt.axhline(y=0, color="r", ls="--")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.grid(axis="both", lw=.5, alpha=.2)
        plt.legend(loc="upper left", fancybox=True)
        plt.xlabel("Time in Months")
        plt.ylabel("Monthly cumulative case change against baseline")
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(7)
        expander_bsts_4 = st.expander("BSTS Cumulative effects")
        expander_bsts_4.write(fig)
        new_df = ci_data.to_csv().encode('utf-8')
        st.download_button(
            "Download BSTS estimates",
            new_df,
            "bsts.csv",
            key="download-csv"
        )

    elif menu_choices == "Interrupted Time Series":
    #if st.sidebar.checkbox("Interrupted Time Series Analyses"):
        st.subheader("ITSA")
        latext = r"""
        ###### We follow [Huitema and Mckean (2007)](https://journals.sagepub.com/doi/abs/10.1177/0013164406294774) and run the following equation:
        $$
        Y_n = \beta_0 + \beta_1 t + \beta_2 D_{t} + \beta_3 [t - T_I] D_{t} + \epsilon_{t}
        $$
        where $Y_t$ represents the outcome. $T_I$ represents the interruption (shock) time. $D_t$ is a dummy variable where $1$ represents the post-intervention
        period. $t$ is time from the start of the series. Note that under this formulation $[t - T_I]$ will be zero for the pre-intervention period,
        and $1, 2, ...., n$ for the post-intervention in equal time intervals. We use heteroskedasticity and
        autocorrelation consistent errors (Newey-West). The interpretation of the estimates is as follows:
        - $\beta_0$ is the intercept
        - $\beta_1$ is the pre-shock trajectory
        - $\beta_2$ is the immediate shock effect
        - $\beta_3$ is the effect of the shock over time i.e. the difference in the pre-shock and post-shock trajectories
        The number of lags in this formulation is calculated for using this formula following Stock and Watson 2007:
        $$
        m = 0.75T^{1/3}
        $$
        The dataset below shows how the ITSA regression is set-up in the backend. For more details, see
        [Ariel Linden, (2015)](https://www.stata-journal.com/article.html?article=st0389)

        It will be great for the scientific community to cite Ariel since this section is
        the Python formulation of his excellent Stata adofile.
        """
        itsa_summary = st.expander("Method summary")
        itsa_summary.write(latext)
        # Estimate needed lags
        df = df[:num_months]
        m = round(0.75 * len(df)**(1/3))
        df["total_time"] = np.arange(len(df))
        df["post_period"] = df["total_time"] - (pre_months - 1)
        df["post_period"] = df["post_period"].clip(lower=0)
        post_obs = df[outcome][pre_months:].sum() #post period total observations
        df.rename(columns={outcome :"outcome"}, inplace=True)
        pre_months2 = pre_months - 2

        df["shock"] = np.where(df.total_time > pre_months2, 1,0)
        df["month"] = pd.DatetimeIndex(df.index).month
        pre_regression = st.expander("ITSA dataset for regression")
        pre_regression.write(df)

        df.reset_index(inplace=True, drop=True)

        model = smf.ols("outcome ~ shock + post_period + total_time + C(month)",
                        data = df).fit(cov_type='HAC', cov_kwds={'maxlags':m})

        model3 = smf.ols("outcome ~ total_time +C(month)", data=df[:pre_months]).fit()
        df4 = df[pre_months:]
        res3 = model3.get_prediction().summary_frame(alpha=.05)
        ypred = model3.get_prediction(df4).summary_frame(alpha=.05)
        res3 = res3.append(ypred, ignore_index=True)
        df2 = model.get_prediction().summary_frame(alpha=.05)

        df2.reset_index(inplace=True, drop=True)
        res3.reset_index(inplace=True, drop=True)
        df.reset_index(inplace=True, drop=True)


        fig, ax = plt.subplots(figsize=(12,5), dpi=150)
        plt.plot(df2.index, df2["mean"], "k-", label="Fitted Observed")

        plt.scatter(df.index, df["outcome"], color="blue", label="Observed")
        plt.plot(res3.index[pre_months:], res3["mean"][pre_months:], color="red", label="Predicted")
        plt.fill_between(res3.index[pre_months:], res3["mean_ci_lower"][pre_months:],
                        res3["mean_ci_upper"][pre_months:], color="green", alpha=.1)
        plt.axvline(pre_months, color="k", ls="--")
        plt.legend()
        plt.grid(alpha=.2);

        model_1 = (model.summary2().tables[0])
        model_2 = (model.summary2().tables[1])
        expander_itsa_1 = st.expander("ITSA Regression Technical Output")
        #expander_itsa.write(model.summary())
        expander_itsa_1.dataframe(model_1)
        expander_itsa_2 = st.expander("ITSA Regression Estimates")
        #expander_itsa.write(model.summary())
        expander_itsa_2.dataframe(model_2)

        itsa_graph = st.expander("ITSA Graph")
        itsa_graph.write(fig)

        shock = model.params["shock"]
        shock_pvalue = model.pvalues["shock"]
        post_month = model.params["post_period"]

        mean_post = df["outcome"][pre_months:].mean()
        mean_pred = res3["mean"][pre_months:].mean()
        total_pred = res3["mean"][pre_months:].sum()

        x = pt.ttest(df["outcome"][pre_months:], res3["mean"][pre_months:], paired=True)
        lower_ci = x["CI95%"][0][0]
        upper_ci = x["CI95%"][0][1]
        p_value = x["p-val"][0]
        a = "statistically significant"
        b = "statistically insignificant"
        p = .05

        itsa_interpretation = f"""
        There were an observed {post_obs:,.1f} {outcome} observations during the post period. Using an interrupted time series approach,
        there would have been a {total_pred:,.1f} expected observations in the post period had the shock not occured. The mean monthly
        observations were {mean_post:,.1f}, while the expected mean monthly observations were {mean_pred:,.1f}. This represents a
        {a if p_value < p else b} mean monthly change of {mean_post - mean_pred:,.1f}[{lower_ci:,.1f}, {upper_ci:,.1f}] or {((mean_post - mean_pred)/mean_pred) * 100:,.2f}%.

        The results of this approach should be viewed with the caveats outlined in the other sections: data quality,
        competing shocks, population composition changes, etc.

        The results will also vary with the phase of the post-period -- that is, the policy impact estimates will vary
        depending on how long the post-period window is. The user of the tool can check this by  changing the end date of the
        time series in the side panel.
        """
        expander_interpretation = st.expander("ITSA summary narrative")
        expander_interpretation.write(itsa_interpretation)

    elif menu_choices == "SARIMAX":
    #if st.sidebar.checkbox("Seasonal Autoregressive Integrated Moving Average"):
        st.subheader("SARIMAX")
        try:
            sar_model = auto_arima(df[outcome][:pre_months], error_action="ignore", trace=False, seasonal=True,
                              suppress_warnings=True, out_of_sample_size=post_months) #Revise this to conform to slider
        except AttributeError:
            error_message = "Data should be float or integer"
        expander_sarima_1 = st.expander("SARIMA Model Selection Automated")
        expander_sarima_1.write(sar_model.summary())
        df1 = df[:-post_months]
        try:
            mod = sm.tsa.SARIMAX(df1[outcome], order=sar_model.order, seasonal_order=sar_model.seasonal_order, trend="c")
            res = mod.fit()
        except AttributeError:
            print("Data should be float or integer")
        expander_sarima_2 = st.expander("SARIMA Model Training Results")
        expander_sarima_2.write(res.summary())
        # Forecast results
        forecast = res.get_forecast(steps = post_months)
        new_df = forecast.summary_frame(alpha=0.05)
        new_df["observed"] = df[outcome][-post_months:]
        new_df = new_df.rename(columns={"mean": "forecast"})
        new_df["differences"] = new_df["observed"] - new_df["forecast"]
        expander_sarima_3 = st.expander("SARIMA Model Forecast Results")
        expander_sarima_3.write(new_df)

        #Summary results
        fig, ax = plt.subplots(figsize=(12,5), dpi=500)
        plt.plot(new_df.index, new_df["forecast"], "k--", label="Forecast")
        plt.plot(new_df.index, new_df["observed"], label="Observed - Post")
        plt.plot(df1.index, df1[outcome], label="Observed - Pre")
        plt.axvline(x=shock_date, ls="--", color="r")
        ax.fill_between(new_df.index, new_df["mean_ci_lower"], new_df["mean_ci_upper"], color="g", alpha=0.1)
        plt.legend(loc="best", fancybox=True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.grid(lw=1, color="blue", alpha=.1)
        expander_sarima_4 = st.expander("SARIMA Model Forecast Graph")
        expander_sarima_4.write(fig)

        cum_sum = np.sum(new_df["differences"])
        mean_diff = np.mean(new_df["differences"])
        narrative = f"The cumulative change in cases under the SARIMAX model was {cum_sum:,.2f} with an average monthly change of {mean_diff:,.2f} cases"
        st.markdown(narrative)
        new_df = new_df.to_csv().encode('utf-8')
        st.download_button(
            "Download SARIMAX estimates",
            new_df,
            "sarimax.csv",
            key="download-csv"
        )

    st_disqus("Suggestions for draft tool")
