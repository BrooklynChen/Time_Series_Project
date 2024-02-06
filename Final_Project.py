import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller, q_stat, acf
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import numpy.linalg as LA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lmfit import minimize, Parameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#%%

# Import Data
energy = pd.read_csv("energy_demand.csv")

# Set the data type of Time_stamp as datetime
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")

# Sort the data and set time as index
energy.sort_values(by='Time_stamp', inplace=True)

energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

print(energy)

# 6. Description of the dataset. Describe the independent variable(s) and dependent variable:
# 6-a. Pre-processing dataset: Dataset cleaning for missing observation
print(f'NA in Energy Dataset:\n{energy.isna().sum()}')

#%%

# 6-b. Plot of the dependent variable versus time.
# Write down your observations.
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot()
ax.plot(energy['kW'], color='green', label='kW', markerfacecolor='#121466')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Smart Meter Energy\n kW versus Date')
plt.legend()
plt.show()

#%%

# 6-c. ACF/PACF of the dependent variable
# Write down your observations

def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.title('ACF/PACF of kW')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

ACF_PACF_Plot(energy['kW'], 20)


#%%

# 6-d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient
# Write down your observations

# Correlation Matrix for kW
corr_metrix = energy.corr()
fig, ax = plt.subplots(figsize=(8, 6))
heatmap1 = sns.heatmap(corr_metrix, cmap ='Blues', annot=True)
heatmap1.set_title('Correlation Matrix for kW', fontdict={'fontsize':18}, pad=16)
plt.show()

#%%
# 6-e. Split the dataset into train set (80%) and test set (20%)

X = energy[['kWh', 'kVAR', 'kVARh']]
y = energy[['kW']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#%%

# 7. Stationarity
# Perform ACF/PACF analysis for stationarity.
# You need to perform ADF-test & kpss-test and plot the rolling mean and variance for the raw data
# and the transformed data

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %.2f" %result[0])
    print('p-value: %.2f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.2f' % (key, value))


def kpss_test(timeseries):
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)


def Cal_rolling_mean_var(x, title):
    rolling_mean = []
    rolling_std = []

    for i in range(0, len(x)):
            rolling_mean.append(np.mean(x[:i+1]))
            rolling_std.append(np.std(x[:i+1]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(rolling_mean)
    ax1.set_title("Rolling Mean -" + title)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Magnitude")

    ax2.plot(rolling_std)
    ax2.set_title("Rolling Variance -" + title)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Magnitude")
    plt.show()

#%%

# For raw data
ACF_PACF_Plot(energy['kW'], 20)

ADF_Cal(energy['kW'])
# If the p-values < 0.05, we reject the null hypothesis. The data sets are stationary.

kpss_test(energy['kW'])
# The p-value of KPSS test is smaller than 0.05, so we can conclude that the data set is not stationary.

Cal_rolling_mean_var(energy['kW'], 'kW')
# The data sets is not stationary,
# since the rolling means and rolling variances does not stabilize once all samples are included.


#%%
# Transformation
# FIRST

diff1 = []
def difference(dataset, interval=1):
    for i in range(interval, len(dataset)):
        val= dataset[i]- dataset[i- interval]
        diff1.append(val)

    plt.plot(diff1)
    plt.grid()
    plt.title("1st order non-seasonal differencing - Energy: kW")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.show()

    ACF_PACF_Plot(diff1, 20)
    ADF_Cal(diff1)
    kpss_test(diff1)
    Cal_rolling_mean_var(diff1, 'kW')


# SECOND

def difference2(dataset, interval=2):
    diff2 = []
    for i in range(interval,len(dataset)):
        val= dataset[i]- 2*dataset[i-interval+1] + dataset[i-interval]
        diff2.append(val)

    plt.plot(diff2)
    plt.grid()
    plt.title("2nd order non-seasonal differencing - Energy: kW")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.show()

    ACF_PACF_Plot(diff2, 20)
    ADF_Cal(diff2)
    kpss_test(diff2)
    Cal_rolling_mean_var(diff2, 'kW')

# THIRD

def difference3(dataset, interval=3):
    diff3 = []
    for i in range(interval,len(dataset)):
        val= dataset[i]- 3*dataset[i-interval+2] + 3*dataset[i-interval+1] - dataset[i-interval]
        diff3.append(val)

    plt.plot(diff3)
    plt.grid()
    plt.title("3rd order non-seasonal differencing - Energy: kW")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.show()

    ACF_PACF_Plot(diff3, 20)
    ADF_Cal(diff3)
    kpss_test(diff3)
    Cal_rolling_mean_var(diff3, 'kW')

#%%
# the transformed data
difference(energy['kW'])

# After performing the first difference transformation of the original raw dataset,
# the rolling mean and standard deviation are approximately horizontal.
# The p-value of ADF Statistic is smaller than the critical value,
# and the p-value of KPSS is greater than 0.05. The dataset is stationary after the transformation.


#%%
# 8. Time series Decomposition: Approximate the trend and the seasonality and plot the detrended and the seasonally adjusted data set using STL method.

# Useing a moving average method to estimate the trend-cycle

def moving_avg_plot(arr, title):

    m = int(input("The order of moving average?"))
    if m % 2 == 0:
        sec_m = int(input("The folding order?"))
        if sec_m % 2 != 0:
            print("The folding order should be even")
        else:
            i = 0
            ma = []
            while i < len(arr) - m:
                ma.append(round(sum(arr[i: i + m]) / m, 3))
                i += 1
            i = 0
            sec_ma = []
            while i < len(arr) - m:
                sec_ma.append(round(sum(ma[i:i + sec_m])/ sec_m, 3))
                i += 1

            detrend_sec = []
            for ind in range(100):
                if ind < sec_m:
                    sec_ma.insert(ind, np.nan)
                    detrend_sec.insert(ind, np.nan)
                else:
                    detrend_sec.append(arr[ind] - sec_ma[ind])

            plt.plot(energy.index[: 100], arr[: 100], label='Original')
            plt.plot(energy.index[: 100], sec_ma[: 100], label=title)
            plt.plot(energy.index[: 100], detrend_sec, label="Detrended Data")
            plt.xticks(rotation=20)
            plt.xlabel('Date')
            plt.ylabel('Magnitude')
            plt.title('Cycle-trend of kW')
            plt.legend()

    else:
        i = 0
        ma = []
        while i < len(arr) - m + 1:
            ma.append(round(sum(arr[i: i+m])/m, 3))
            i += 1

        for ind in range(int((m - 1)/2)):
            ma.insert(ind, np.nan)

        detrend = []
        for ind in range(100):
            if ind < int((m - 1)/2):
                ma.insert(ind, np.nan)
                detrend.insert(ind, np.nan)
            else:
                detrend.append(arr[ind] - ma[ind])

        plt.plot(energy.index[: 100], arr[: 100], label='Original')
        plt.plot(energy.index[: 100], ma[: 100], label=title)
        plt.plot(energy.index[: 100], detrend, label="Detrended Data")
        plt.xticks(rotation=20)
        plt.xlabel('Date')
        plt.ylabel('Magnitude')
        plt.title('Cycle-trend of kW')
        plt.legend()


# fig = plt.figure(figsize=(16, 8))
# fig.add_subplot(221)
# moving_avg_plot(diff1, '3-MA')
# fig.add_subplot(222)
# moving_avg_plot(diff1, '5-MA')
# fig.add_subplot(223)
# moving_avg_plot(diff1, '7-MA')
# fig.add_subplot(224)
# moving_avg_plot(diff1, '9-MA')
#
# plt.tight_layout()
# plt.show()

#%%

fig = plt.figure(figsize=(14, 8))
fig.add_subplot(221)
moving_avg_plot(diff1, '2x4-MA')
fig.add_subplot(222)
moving_avg_plot(diff1, '2x6-MA')
fig.add_subplot(223)
moving_avg_plot(diff1, '2x8-MA')
fig.add_subplot(224)
moving_avg_plot(diff1, '2x10-MA')

plt.tight_layout()
plt.show()

#%%
energy = pd.read_csv("energy_demand.csv")
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)

diff_kW = pd.DataFrame({'kW': diff1})
diff_kW.index = energy["Time_stamp"][6:5005]

energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)
STL = STL(diff_kW[:500], period=48)
res = STL.fit()
fig = res.plot()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=(12, 8))
plt.plot(T, label='Trend')
plt.plot(S, label='Seasonal')
plt.plot(R, label='Residuals')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Cycle-trend of kW')
plt.legend()
plt.show()

#%%

# Calculate the seasonally adjusted data and plot it versus the original data

seasonal_adj = diff_kW['kW'][:500] - S
# print(seasonal_adj)

plt.figure(figsize=(12, 8))
plt.plot(seasonal_adj, label='Seasonal Adjustment')
plt.plot(diff_kW['kW'][:500], label='Original Data')
# plt.xticks(rotation=20)
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Seasonal Adjustment of kW')
plt.legend()
plt.show()

#%%

# Find the out the strength of the trend and seasonality
# Refer to the lecture notes for different type of time series decomposition techniques

# strength of trend
Ft = 1-(np.var(R)/np.var(T+R))
print(f'The strength of trend for this data set is {round(Ft, 3)}')

# strength of seasonality
Fs = 1-(np.var(R)/np.var(S+R))
print(f'The strength of seasonality for this data set is {round(Fs, 3)}')

# In this dataset, since the strength of seasonality is 0.6156, this dataset is seasonal. On the other hand, the strength of trend is close to zero, so the trend of the dataset is weak.
# After first-order differencing, the strength of its trend and seasonality became weak.

#%%

# 9. Using the Holt-Winters method try to find the best fit using the train dataset and make a prediction using the test set

energy = pd.read_csv("energy_demand.csv")
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)

energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

energy1 = energy.copy()
energy1['kW'] += 0.01

y1 = energy1[['kW']]
yt, yf = train_test_split(y1, shuffle= False, test_size=0.2)

holtt = ets.ExponentialSmoothing(yt, trend='mul', damped_trend=True,seasonal='mul', seasonal_periods=49).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print(f'Mean square error for Holt-Winter is {MSE:.2f}')

plt.rc("figure", figsize=(10, 8))
plt.rc("font", size=12)
fig, ax = plt.subplots()
ax.plot(yt, label= "Train Data")
ax.plot(yf, label= "Test Data")
ax.plot(holtf,label= "Holt-Winter")
plt.legend(loc='upper left')
plt.title(f'Holt-Winter- MSE = {MSE:.2f}')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.xticks(rotation=20)
plt.grid()
plt.show()

residuals = yf.values.flatten() - holtf.values.flatten()
residual_mean = residuals.mean()
residual_var = residuals.var()

print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')
#%%
# 10. Feature Selection

energy = pd.read_csv("energy_demand.csv")
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)
energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

# Backward Stepwise Regression
X = energy[['kVAR', 'kVARh', 'kWh']]
y = energy[['kW']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_train = sm.add_constant(X_train)
model_bsr = sm.OLS(y_train, X_train).fit()
full_model_aic = model_bsr.aic
full_model_bic = model_bsr.bic
full_model_adj_r2 = model_bsr.rsquared_adj

while True:
    m_pvalue = model_bsr.pvalues.idxmax()
    x_train = X_train.drop(m_pvalue, axis=1)

    model_bsr = sm.OLS(y_train, X_train).fit()

    reduced_model_aic = model_bsr.aic
    reduced_model_bic = model_bsr.bic
    reduced_model_adj_r2 = model_bsr.rsquared_adj

    if reduced_model_aic < full_model_aic and reduced_model_bic < full_model_bic and reduced_model_adj_r2 > full_model_adj_r2:
        full_model_aic = reduced_model_aic
        full_model_bic = reduced_model_bic
        full_model_adj_r2 = reduced_model_adj_r2
    else:
        break

print(x_train.columns)
print(f'AIC:{model_bsr.aic:.2f}')
print(f'BIC:{model_bsr.bic:.2f}')
print(f'Adjusted R-squared::{model_bsr.rsquared_adj:.2f}')

# The features 'kWh' and 'kVAR' will recommend keeping and the other features will recommend eliminating

# SVD Analysis (collinearity detection)
s, d, v = LA.svd(X[['kVAR', 'kWh']])
print('Singular Values=', d)

# Calculate the condition number
k = LA.cond(X[['kVAR', 'kWh']])
print('k=', round(k, 4))
# Since the κ > 1000, the degree of co-linearity is severe.

#%%
# VIF method
energy = pd.read_csv("energy_demand.csv")
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)
energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

X = energy[['kVAR', 'kVARh', 'kWh']]
y = energy[['kW']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

num_feats = len(X_train.columns)

while num_feats > 0:

    vif = pd.DataFrame()
    vif['variable'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

    if vif['VIF'].max() < 5:
        break
    else:
        feat_remove = vif.loc[vif['VIF'].idxmax(), 'variable']
        X_train = X_train.drop(feat_remove, axis=1)
        num_feats = len(X_train.columns)

model_vif = sm.OLS(y_train, X_train).fit()

print(vif)
print(f'Selected features:{X_train.columns.tolist()}')
print(f'AIC:{model_vif.aic:.2f}')
print(f'BIC:{model_vif.bic:.2f}')
print(f'Adjusted R-squared:{model_vif.rsquared_adj:.2f}')

# The features 'kVARh' and 'kVAR' will recommend keeping and the other features will recommend eliminating

# SVD Analysis (collinearity detection)
s, d, v = LA.svd(X[['kVAR', 'kVARh']])
print(f'Singular Values={d}')

# Calculate the condition number
k = LA.cond(X[['kVAR', 'kVARh']])
print(f'k={k:.2f}')

#%%

# feature selection - PCA

X = energy[['kWh', 'kVAR', 'kVARh']]
y = energy[['kW']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute the covariance matrix and eigenvalues/eigenvectors
covariance_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and eigenvectors in descending order of eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Select the number of principal components to keep based on explained variance
explained_variance = eigenvalues / sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)
num_components = 0
for i, variance in enumerate(cumulative_variance):
    if variance >= 0.95:
        num_components = i + 1
        break

# Project data onto principal components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_scaled)

# Select features with highest loadings in each principal component
feature_loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i}" for i in range(1, num_components + 1)],
                                index=X.columns)
top_features = []
for pc in feature_loadings.columns:
    top_feature = feature_loadings[pc].abs().idxmax()
    top_features.append(top_feature)

# Print top features and their loadings
print("Top features:")
for i, feature in enumerate(top_features):
    loading = feature_loadings.loc[feature, f"PC{i + 1}"]
    print(f"{feature} (loading = {loading:.2f})")

# Plot explained variance by number of principal components

plt.plot(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance")
plt.title("PCA - Explained Variance")
plt.show()

#%%

X = energy[['kWh', 'kVAR', 'kVARh']]
y = energy[['kW']]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Create a PCA object and fit it to the normalized data
pca = PCA()
pca.fit(X_norm)

# Get the explained variance of each principal component
explained_var = pca.explained_variance_ratio_

# Plot the cumulative explained variance to determine the number of components to keep
cumulative_var = np.cumsum(explained_var)
plt.plot(cumulative_var)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title("PCA - Explained Variance")
plt.show()

# Choose the number of components to keep based on the plot
n_components = 2

# Transform the data using the selected number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_norm)

# Use the transformed data for regression analysis
reg = LinearRegression().fit(X_pca, y)
y_pred = reg.predict(X_pca)

RSS = ((y - y_pred) ** 2).sum()
n_samples, n_components = X_pca.shape
df = n_samples - n_components - 1

AIC = n_samples * np.log(RSS / n_samples) + 2 * (n_components + 1)
BIC = n_samples * np.log(RSS / n_samples) + np.log(n_samples) * (n_components + 1)
R2 = reg.score(X_pca, y)
adj_R2 = 1 - (1 - R2) * (n_samples - 1) / (n_samples - n_components - 1)
print(f'coefficients:{reg.coef_}')
print(f'intercept:{reg.intercept_}')
print(f'prediction:{y_pred}')
print('AIC:', AIC)
print('BIC:', BIC)
print('Adjusted R-squared:', adj_R2)
#%%

# Simple Linear Regression

X = energy[['kVAR']]
y = energy[['kW']]

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

print(f'AIC:{results.aic:.2f}')
print(f'BIC:{results.bic:.2f}')
print(f'Adjusted R-squared:{results.rsquared_adj:.2f}')

#%%
# 11. Base-models: average, naïve, drift, simple and exponential smoothing.
# You need to perform an h-step prediction based on the base models and
# compare the SARIMA model performance with the base model predication.

# SARIMA model performance

energy = pd.read_csv("energy_demand.csv")

energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)
energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

train, test = train_test_split(energy['kW'], shuffle=False, test_size=0.2)

sarima = SARIMAX(train, order=(2, 0, 2), seasonal_order=(1, 0, 0, 48))
model_fit = sarima.fit()
predictions = model_fit.forecast(steps=len(test))

plt.figure(figsize=(12, 8))
plt.plot(train, label="Training")
plt.plot(test, label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.title('SARIMA model predictions', fontsize=20)
plt.ylabel('kW', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xticks(rotation=20)
plt.legend()
plt.show()

#%%

energy = pd.read_csv("energy_demand.csv")
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")

energy.sort_values(by='Time_stamp', inplace=True)
energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

# 11-a. Average
y = energy[['kW']]

y_train, y_test = train_test_split(y, test_size=0.2, random_state=42, shuffle=False)

y_hat_train = y_train.copy()
temp = []
for t in range(len(y_train)):
    if t < 1:
        temp.append('NA')
    elif t == 1:
        temp.append(y_train['kW'][0])
    else:
        temp.append(np.mean(y_train['kW'][:t]))
y_hat_train.insert(1, 'avg_forecast', temp)

n = len(y_train)
y_hat_test = y_test.copy()
y_hat_test['avg_forecast'] = np.mean(y_train['kW'][:n])

frames = [y_hat_train, y_hat_test]
df = pd.concat(frames)
print(df)

residuals = y_test.values - y_hat_test['avg_forecast'].values
residual_mean = residuals.mean()
residual_var = residuals.var()
rmse = np.sqrt(np.mean(np.square(y_test.values - y_hat_test['avg_forecast'].values)))

print(f'RMSE: {rmse:.4f}')
print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')

# Plot the test set, training set and the h-step forecast in one graph
plt.figure(figsize=(12, 8))
plt.plot(y_train['kW'], label='Train')
plt.plot(y_test['kW'], label='Test')
plt.plot(y_hat_test['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.title('Average Forecast Method')
plt.xticks(rotation=20)
plt.xlabel('Date')
plt.ylabel('kW')
plt.show()

#%%

# 11-b. Naïve

y = energy[['kW']]
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42, shuffle=False)

y_hat_train = y_train.copy()
temp2 = []
for t in range(len(y_train)):
    if t < 1:
        temp2.append('NA')
    else:
        temp2.append(y_train['kW'][t-1])
y_hat_train.insert(1, 'nai_forecast', temp2)

y_hat_test = y_test.copy()
n = len(y_train)
y_hat_test['nai_forecast'] = y_train['kW'][n-1]

frames = [y_hat_train, y_hat_test]
df2 = pd.concat(frames)
print(df2)

residuals = y_test.values - y_hat_test['nai_forecast'].values
residual_mean = residuals.mean()
residual_var = residuals.var()
rmse = np.sqrt(np.mean(np.square(y_test.values - y_hat_test['nai_forecast'].values)))

print(f'RMSE: {rmse:.4f}')
print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')

# Plot the test set, training set and the h-step forecast in one graph

plt.figure(figsize=(12, 8))
plt.plot(y_train['kW'], label='Train')
plt.plot(y_test['kW'], label='Test')
plt.plot(y_hat_test['nai_forecast'], label='Naïve Forecast')
plt.legend(loc='best')
plt.title('Naïve Forecast Method')
plt.xticks(rotation=20)
plt.xlabel('Date')
plt.ylabel('kW')
plt.show()

#%%

# 11-c. Drift

y = energy[['kW']]
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42, shuffle=False)

y_hat_train = y_train.copy()
temp3 = []
for t in range(len(y_train)):
    if t < 2:
        temp3.append('NA')
    else:
        temp3.append(y_train['kW'][t-1]+((y_train['kW'][t-1]-y_train['kW'][0])/(t-1)))
y_hat_train.insert(1, 'dri_forecast', temp3)

y_hat_test = y_test.copy()
temp4 = []
m = len(y_train) - 1

for t in range(len(y_test)):
    temp4.append(y_train['kW'][m]+((y_train['kW'][m]-y_train['kW'][0])/(m))*(t+1))
y_hat_test.insert(1, 'dri_forecast', temp4)

frames = [y_hat_train, y_hat_test]
df3 = pd.concat(frames)
print(df3)

residuals = y_test['kW'].values - y_hat_test['dri_forecast'].values
residual_mean = residuals.mean()
residual_var = residuals.var()
rmse = np.sqrt(np.mean(np.square(y_test['kW'].values - y_hat_test['dri_forecast'].values)))

print(f'RMSE: {rmse:.4f}')
print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')

# Plot the test set, training set and the h-step forecast in one graph

plt.figure(figsize=(12, 8))
plt.plot(y_train['kW'], label='Train')
plt.plot(y_test['kW'], label='Test')
plt.plot(y_hat_test['dri_forecast'], label='Drift Forecast')
plt.legend(loc='best')
plt.title('Drift Method')
plt.xticks(rotation=20)
plt.xlabel('Date')
plt.ylabel('kW')
plt.show()

#%%

# 11-d. SES

df5 = energy[['kW']]
y_train, y_test = train_test_split(df5, test_size=0.2, random_state=42, shuffle=False)

def SES_plot(alfa, column_name, title):
    temp7 = []
    n = len(y_train)
    for t in range(len(df5)):
        if t < 1:
            temp7.append('NA')
        elif t == 1:
            temp7.append(alfa * df5['kW'][0] + (1-alfa) * df5['kW'][0])
        else:
            temp7.append(alfa * df5['kW'][t-1] + (1-alfa) * temp7[t-1])
    df5.insert(1, column_name, temp7)

    plt.plot(y_train, label='Train')
    plt.plot(y_test, label='Test')
    plt.plot(df5[column_name][n:], label='SES')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xticks(rotation=20)
    plt.xlabel('Date')
    plt.ylabel('kW')

    residuals = y_test.values - df5[column_name][n:].values
    residual_mean = residuals.mean()
    residual_var = residuals.var()
    rmse = np.sqrt(np.mean(np.square(y_test.values - df5[column_name][n:].values)))

    print(f'RMSE: {rmse:.4f}')
    print(f'Residual Mean: {residual_mean:.4f}')
    print(f'Residual Variance: {residual_var:.4f}')

fig = plt.figure(figsize=(12,8))
fig.add_subplot(221)
SES_plot(0, 'forecast_alfa=0', 'SES method: alfa=0')
fig.add_subplot(222)
SES_plot(0.25, 'forecast_alfa=0.25', 'SES method: alfa=0.25')
fig.add_subplot(223)
SES_plot(0.75, 'forecast_alfa=0.75', 'SES method: alfa=0.75')
fig.add_subplot(224)
SES_plot(0.99, 'forecast_alfa=0.99', 'SES method: alfa=0.99')

plt.tight_layout()
plt.show()



#%%

# 12- Develop the multiple linear regression model that represent the dataset.
# Check the accuracy of the developed model.
# a. You need to include the complete regression analysis into your report.
# Perform one-step ahead prediction and compare the performance versus the test set.
# b. Hypothesis tests analysis: F-test, t-test.
# c. AIC, BIC, RMSE, R-squared and Adjusted R-squared
# d. ACF of residuals.
# e. Q-value
# f. Variance and mean of the residuals.

X = energy[['kVAR', 'kVARh']]
y = energy[['kW']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False)

# Create a multiple linear regression model using the training data
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()

# Evaluate the performance of the model using the testing data
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

# Conduct hypothesis tests to determine the significance of the regression coefficients
print(model.summary())

# Calculate evaluation metrics such as AIC, BIC, RMSE, R-squared, and adjusted R-squared
RSS = ((y_test - y_pred) ** 2).sum()
n_samples, n_features = X_test.shape
df = n_samples - n_features - 1
AIC = n_samples * np.log(RSS / n_samples) + 2 * (n_features + 1)
BIC = n_samples * np.log(RSS / n_samples) + np.log(n_samples) * (n_features + 1)
R2 = model.rsquared
adj_R2 = 1 - (1 - R2) * (n_samples - 1) / (n_samples - n_features - 1)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print('AIC:', AIC)
print('BIC:', BIC)
print('RMSE:', RMSE)
print('R-squared:', R2)
print('Adjusted R-squared:', adj_R2)
#%%

# Simple Linear Regression
X = energy[['kVAR']]
y = energy[['kW']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False)

# Create a multiple linear regression model using the training data
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()

# Evaluate the performance of the model using the testing data
X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

# Conduct hypothesis tests to determine the significance of the regression coefficients
print(model.summary())

# Calculate evaluation metrics such as AIC, BIC, RMSE, R-squared, and adjusted R-squared
RSS = ((y_test - y_pred) ** 2).sum()
n_samples, n_features = X_test.shape
df = n_samples - n_features - 1
AIC = n_samples * np.log(RSS / n_samples) + 2 * (n_features + 1)
BIC = n_samples * np.log(RSS / n_samples) + np.log(n_samples) * (n_features + 1)
R2 = model.rsquared
adj_R2 = 1 - (1 - R2) * (n_samples - 1) / (n_samples - n_features - 1)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print('AIC:', AIC)
print('BIC:', BIC)
print('RMSE:', RMSE)
print('R-squared:', R2)
print('Adjusted R-squared:', adj_R2)

#%%

# Step 6: Check the autocorrelation function (ACF) of the residuals to see if they exhibit any pattern
residuals = y_test.values.ravel() - y_pred.ravel()
plot_acf(np.squeeze(residuals))

# Step 7: Calculate the Q-value to test for autocorrelation in the residuals

acor = []
def autocor(data, lags, title):
    mean = sum(data) / len(data)
    var = sum([(s - mean) ** 2 for s in data]) / len(data)
    ndata = [s - mean for s in data]

    for l in lags:
        c = 1
        if (l != 0):
            l = abs(l)
            tmp = [ndata[l:][i] * ndata[:-l][i]
                   for i in range(len(data) - l)]

            c = sum(tmp) / len(data) / var
        acor.append(c)

lags = 20
autocor(residuals, range(-20, 21), 'Autocorrelation Function of Residuals')

Q = len(y)*np.sum(np.square(acor[lags:]))
n = len(y)
p = X_train.shape[1] - 1
DOF = n - p - 1
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)

if Q < chi_critical:
    print('The residual is white')
else:
    print('The residual is NOT white')

print(sm.stats.acorr_ljungbox(residuals, lags=[lags]))

# Step 8: Calculate the variance and mean of the residuals to check for heteroscedasticity
residuals_var = residuals.var()
residuals_mean = residuals.mean()
print('Residuals variance:', residuals_var)
print('Residuals mean:', residuals_mean)

# 12-b. Hypothesis tests analysis: F-test, t-test

tt = model.t_test(np.eye(len(model.params)))
print(tt.summary())

A = np.identity(len(model.params))
A = A[1:, :]
print(model.f_test(A))

# The t-test statistic helps to determine the correlation between the response and the predictor variables. Since the p-values for the t-tests are less than the significance level (0.05), then we can reject the null hypothesis that the corresponding coefficient is zero.
# This indicates that the predictor variable is significantly associated with the response variable.
# Since the p-value for the F-test is less than the significance level, we can reject the null hypothesis that all the coefficients in the model are zero. This indicates that at least one of the predictor variables is significantly associated with the response variable.
# The t-test and F-test both show statistical significance, so we can conclude that the model is a good fit for the data and the predictor variables are significant in explaining the variation in the response variable.

#%%

# 13- ARMA and ARIMA and SARIMA model order determination: Develop an ARMA, ARIMA and SARIMA model that represent the dataset.
# a. Preliminary model development procedures and results. (ARMA model order determination). Pick at least two orders using GPAC table.
# b. Include a plot of the autocorrelation function and the GPAC table within this section
# c. Include the GPAC table in your report and highlight the estimated order

energy = pd.read_csv("energy_demand.csv")

energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)
energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

# Find AR and MA oder
aic = []
bic = []
xtick_labels = []

for i in range(1, 5):
    for j in range(0, 3):
        print(f"Fitting ARIMA({i}, 0, {j}) model...")
        res = sm.tsa.ARIMA(energy['kW'], order=(i, 0, j)).fit()
        print(f"AIC: {res.aic}, BIC: {res.bic}")
        aic.append(res.aic)
        bic.append(res.bic)
        xtick_labels.append(f'({i}, 0, {j})')

plt.figure(figsize=(12, 10))
plt.plot(aic, label='AIC')
plt.plot(bic, label='BIC')
plt.legend()
plt.title('AIC and BIC of ARIMA Model')
plt.xlabel('ARIMA')
plt.ylabel('Magnitude')
plt.xticks(range(len(xtick_labels)), xtick_labels, rotation=45)
plt.show()

#%%

# ARIMA
res = sm.tsa.ARIMA(energy['kW'], order=(2, 0, 2)).fit()
print(round(res.params, 3))

#%%

def GPAC():
    np.random.seed(6313)
    N = int(input('Enter the number of data samples: '))
    mean_e = float(input('Enter the mean of white noise: '))
    var_e = float(input('Enter the variance of the white noise: '))

    na = int(input('Enter AR order: '))
    nb = int(input('Enter MA order: '))

    arparams = np.array(eval(input('Enter the coefficients of AR: ex. 1, -0.1, 0')))
    maparams = np.array(eval(input('Enter the coefficients of MA: ex. 1, 0.3, 0.6')))

    ar = np.r_[arparams]
    ma = np.r_[maparams]

    arma_process = sm.tsa.ArmaProcess(ar, ma)

    def s(t):
        if t == 0:
            return 1
        else:
            return 0

    def g(t):
        if t < 0:
            return 0
        elif t == 0:
            return ma[0] * s(0)

        # ARMA(1,0)
        if na == 1 and nb == 0:
            return -ar[1] * g(t-1)

        # ARMA(1,1)
        if na == nb == 1:
            if t == 1:
                return ma[1] * s(0) - ar[1] * g(0)
            else:
                return - ar[1] * g(t - 1)

        # ARMA(1,2)
        if na == 1 and nb == 2:
            if t == 1:
                return ma[1] * s(0) - ar[1] * g(0)
            elif t == 2:
                return ma[2] * s(0) - ar[1] * g(1)
            else:
                return - ar[1] * g(t - 1)

        # ARMA(0,1)
        if na == 0 and nb == 1:
            return ma[1] * s(t - 1)

        # ARMA(0,2)
        if na == 0 and nb == 2:
            if t == 0 or t == 1 or t == 2:
                return ma[t] * s(0)
            else:
                return 0

        # ARMA(0,3)
        if na == 0 and nb == 3:
            if t == 0 or t == 1 or t == 2 or t == 3:
                return ma[t] * s(0)
            else:
                return 0

        # ARMA(2,0)
        if na == 2 and nb == 0:
            return -ar[1] * g(t-1)

        # ARMA(2,1)
        if na == 2 and nb == 1:
            if t == 1:
                return -ar[1] * g(0)
            else:
                return -ar[1] * g(t-1) - ar[2] * g(t-2)

        # ARMA(2,2)
        elif na == nb == 2:
            if t == 1:
                return ma[1] * s(0) - ar[1] * g(0)
            elif t == 2:
                return ma[2] * s(t - 2) - ar[1] * g(t - 1) - ar[2] * g(t - 2)
            else:
                return - ar[1] * g(t - 1) - ar[2] * g(t - 2)

    def Rye(t):
        if t <= 0:
            return g(-t) * var_e
        else:
            return 0


    def Ry(t):
        # ARMA(2, 2)
        if na == nb == 2:
            a = np.array([
                [ar[0], ar[1], ar[2]],
                [ar[1], ar[0]+ar[2], 0],
                [ar[2], ar[1], ar[0]],
            ])

            b = np.array([Rye(0)+ma[1]*Rye(-1)+ma[2]*Rye(-2),
                          Rye(1)+ma[1]*Rye(0)+ma[2]*Rye(-1),
                          Rye(2)+ma[1]*Rye(1)+ma[2]*Rye(0)]).reshape(3, 1)
            a_inv = np.linalg.inv(a)
            ans = a_inv.dot(b)
            if t == 0:
                return ans[0]
            elif t == 1:
                return ans[1]
            elif t == 2:
                return ans[2]
            else:
                return Rye(t)+ma[1]*Rye(t-1)+ma[2]*Rye(t-2)-ar[1]*Ry(t-1)-ar[2]*Ry(t-2)
        # ARMA(2, 1)
        if na == 2 and nb == 1:
            a = np.array([
                [ar[0], ar[1], ar[2]],
                [ar[1], ar[0]+ar[2], 0],
                [ar[2], ar[1], ar[0]],
            ])

            b = np.array([Rye(0)+ma[1]*Rye(-1),
                          Rye(1)+ma[1]*Rye(0),
                          Rye(2)+ma[1]*Rye(1)]).reshape(3, 1)
            a_inv = np.linalg.inv(a)
            ans = a_inv.dot(b)
            if t == 0:
                return ans[0]
            elif t == 1:
                return ans[1]
            elif t == 2:
                return ans[2]
            else:
                return Rye(t)+ma[1]*Rye(t-1)-ar[1]*Ry(t-1)-ar[2]*Ry(t-2)

        # ARMA(2, 0)
        if na == 2 and nb == 0:
            a = np.array([
                [ar[0], ar[1], ar[2]],
                [ar[1], ar[0]+ar[2], 0],
                [ar[2], ar[1], ar[0]],
            ])

            b = np.array([Rye(0), Rye(1), Rye(2)]).reshape(3, 1)
            a_inv = np.linalg.inv(a)
            ans = a_inv.dot(b)
            if t == 0:
                return ans[0]
            elif t == 1:
                return ans[1]
            elif t == 2:
                return ans[2]
            else:
                return Rye(t)-ar[1]*Ry(t-1)-ar[2]*Ry(t-2)

        # ARMA(1, 0)
        elif na == 1 and nb == 0:
            a = np.array([
                [ar[0], ar[1]],
                [ar[1], ar[0]]])
            b = np.array([Rye(0)+ma[1]*Rye(-1),
                          0]).reshape(2, 1)
            a_inv = np.linalg.inv(a)
            ans = a_inv.dot(b)
            if t == 0:
                return ans[0]
            elif t == 1:
                return ans[1]
            else:
                return -ar[1]*Ry(t-1)

        # ARMA(1, 1)
        elif na == nb == 1:
            a = np.array([
                [ar[0], ar[1]],
                [ar[1], ar[0]]])
            b = np.array([Rye(0)+ma[1]*Rye(-1),
                          Rye(1)+ma[1]*Rye(0)]).reshape(2, 1)
            a_inv = np.linalg.inv(a)
            ans = a_inv.dot(b)
            if t == 0:
                return ans[0]
            elif t == 1:
                return ans[1]
            else:
                return -ar[1]*Ry(t-1)

        # ARMA(1, 2)
        if na == 1 and nb == 2:
            a = np.array([
                [ar[0], ar[1], 0],
                [ar[1], ar[0], 0],
                [0, ar[1], ar[0]],
            ])

            b = np.array([Rye(0)+ma[1]*Rye(-1)+ma[2]*Rye(-2),
                          Rye(1)+ma[1]*Rye(0)+ma[2]*Rye(-1),
                          Rye(2)+ma[1]*Rye(1)+ma[2]*Rye(0)]).reshape(3, 1)
            a_inv = np.linalg.inv(a)
            ans = a_inv.dot(b)
            if t == 0:
                return ans[0]
            elif t == 1:
                return ans[1]
            elif t == 2:
                return ans[2]
            else:
                return Rye(t)+ma[1]*Rye(t-1)+ma[2]*Rye(t-2)-ar[1]*Ry(t-1)

        # ARMA(0, 1)
        elif na == 0 and nb == 1:
            return np.array([ma[0]*Rye(t)+ma[1]*Rye(t-1)])

        # ARMA(0, 2)
        elif na == 0 and nb == 2:
            return np.array([ma[0]*Rye(t)+ma[1]*Rye(t-1)+ma[2]*Rye(t-2)])

        # ARMA(0, 3)
        elif na == 0 and nb == 3:
            return np.array([ma[0]*Rye(t)+ma[1]*Rye(t-1)+ma[2]*Rye(t-2)+ma[3]*Rye(t-3)])

    # print('g:', g(0), g(1), g(2), g(3), g(4), g(5))
    # print('Ry:', Ry(0), Ry(1), Ry(2), Ry(3), Ry(4), Ry(5))

    data = []
    for j in range(0, 7):
        row = []
        for k in range(1, 7):
            if k == 1:
                gp = float(Ry(k+j)/Ry(j))
            elif k == 2:
                gp = np.linalg.det(np.stack((np.concatenate((Ry(j), Ry(j + 1)))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k)))))) / np.linalg.det(
                    np.stack((np.concatenate((Ry(j), Ry(abs(j - k + 1))))
                              , np.concatenate((Ry(j + k - 1), Ry(j))))))
            elif k == 3:
                gp = np.linalg.det(np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(j + 1)))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(j + 2)))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k -2), Ry(j + k)))))) / np.linalg.det(
                    np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - k + 1))))
                              , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - k + 2))))
                              , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j))))))
            elif k == 4:
                gp = np.linalg.det(np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(j + 1)))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(j + 2)))
                                        , np.concatenate((Ry(j + 2), Ry(j + 1), Ry(j), Ry(j + 3)))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j + k - 3), Ry(j + k)))))) / np.linalg.det(
                            np.stack((np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - k + 1))))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - k + 2))))
                                        , np.concatenate((Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - k + 3))))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j + k - 3), Ry(j))))))))

            elif k == 5:
                gp = np.linalg.det(np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - 3)), Ry(j + 1)))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(j + 2)))
                                        , np.concatenate((Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(j + 3)))
                                        , np.concatenate((Ry(j + 3), Ry(j + 2), Ry(j + 1), Ry(j), Ry(j + 4)))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j + k - 3), Ry(j + k - 4), Ry(j + k)))))) / np.linalg.det(
                            np.stack((np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - 3)), Ry(abs(j - k + 1))))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - k + 2))))
                                        , np.concatenate((Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - k + 3))))
                                        , np.concatenate((Ry(j + 3), Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - k + 4))))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j + k - 3), Ry(j + k - 4), Ry(j))))))))
            elif k == 6:
                gp = np.linalg.det(np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - 3)), Ry(abs(j - 4)), Ry(j + 1)))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - 3)), Ry(j + 2)))
                                        , np.concatenate((Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(j + 3)))
                                        , np.concatenate((Ry(j + 3), Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(j + 4)))
                                        , np.concatenate((Ry(j + 4), Ry(j + 3), Ry(j + 2), Ry(j + 1), Ry(j), Ry(j + 5)))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j + k - 3), Ry(j + k - 4), Ry(j + k - 5), Ry(j + k)))))) / np.linalg.det(
                            np.stack((np.stack((np.concatenate((Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - 3)), Ry(abs(j - 4)), Ry(abs(j - k + 1))))
                                        , np.concatenate((Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - 3)), Ry(abs(j - k + 2))))
                                        , np.concatenate((Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - 2)), Ry(abs(j - k + 3))))
                                        , np.concatenate((Ry(j + 3), Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - 1)), Ry(abs(j - k + 4))))
                                        , np.concatenate((Ry(j + 4), Ry(j + 3), Ry(j + 2), Ry(j + 1), Ry(j), Ry(abs(j - k + 5))))
                                        , np.concatenate((Ry(j + k - 1), Ry(j + k - 2), Ry(j + k - 3), Ry(j + k - 4), Ry(j + k - 5), Ry(j))))))))


            row.append(round(gp, 2))
        data.append(row)

    df = pd.DataFrame(data, index=range(0, 7), columns=range(1, 7))
    print(df)
    sns.heatmap(df, annot=True, annot_kws={"size": 7})
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.show()

GPAC()

#%%
# ARMA
res = sm.tsa.ARIMA(energy['kW'], order=(2, 0, 2)).fit()
print(res.summary())

# Print the parameter estimates
print(res.params)
# Print the standard deviation of the parameter estimates
print(res.bse)
# Print the confidence intervals
print(res.conf_int())

residuals = res.resid

residual_mean = residuals.mean()
residual_var = residuals.var()

pred = res.predict(start='2018-03-05 09:00:00', end='2018-06-22 15:30:00')
actual = energy.loc['2018-03-05 09:00:00':'2018-06-22 15:30:00', 'kW']
rmse = np.sqrt(mean_squared_error(actual, pred))

print(f"RMSE: {rmse:.2f}")
print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')

forecast = res.predict(start=len(y), end=len(y)+4)
print(forecast)

#%%
# ARIMA
res = sm.tsa.ARIMA(energy['kW'], order=(2, 1, 2)).fit()
print(res.summary())

# Print the parameter estimates
print(res.params)
# Print the standard deviation of the parameter estimates
print(res.bse)
# Print the confidence intervals
print(res.conf_int())

residuals = res.resid

residual_mean = residuals.mean()
residual_var = residuals.var()

pred = res.predict(start='2018-03-05 09:00:00', end='2018-06-22 15:30:00')
actual = energy.loc['2018-03-05 09:00:00':'2018-06-22 15:30:00', 'kW']
rmse = np.sqrt(mean_squared_error(actual, pred))

print(f"RMSE: {rmse:.2f}")

print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')
#%%
# SARIMA
res = sm.tsa.SARIMAX(energy['kW'], order=(2, 0, 2), seasonal_order=(1, 0, 0, 48)).fit()
print(res.summary())

# Print the parameter estimates
print(res.params)
# Print the standard deviation of the parameter estimates
print(res.bse)
# Print the confidence intervals
print(res.conf_int())

residuals = res.resid

residual_mean = residuals.mean()
residual_var = residuals.var()

pred = res.predict(start='2018-03-05 09:00:00', end='2018-06-22 15:30:00')
actual = energy.loc['2018-03-05 09:00:00':'2018-06-22 15:30:00', 'kW']
rmse = np.sqrt(mean_squared_error(actual, pred))

print(f"RMSE: {rmse:.2f}")

print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')
#%%

# Q14-Estimate ARMA model parameters using the Levenberg Marquardt algorithm.
# Display the parameter estimates, the standard deviation of the parameter estimates and confidence intervals.

# Define the objective function to be minimized
def objective(params):
    order = (int(params['p'].value), 0, int(params['q'].value))
    model = sm.tsa.ARIMA(energy['kW'], order=order)
    results = model.fit()
    y_pred = results.predict()
    return energy['kW'] - y_pred

# Set the initial parameter values and bounds
params = Parameters()
params.add('p', value=1, min=0, max=4)
params.add('q', value=1, min=0, max=4)

# Run the optimization
result = minimize(objective, params, method='leastsq')

# Get the estimated parameter values
p = result.params['p'].value
q = result.params['q'].value
print(f"ARMA({p},{q}) model parameters: p={p}, q={q}")

model = sm.tsa.ARIMA(energy['kW'], order=(p, 0, q))
results = model.fit()
# Print the parameter estimates
print(f'Parameter Estimates:\n{res.params}')
# Print the standard deviation of the parameter estimates
print(f'Standard Deviation:\n{res.bse}')
# Print the confidence intervals
print(f'Confidence Interval:\n{res.conf_int()}')


#%%
# 15. Deep Learning Model: Fit the dataset into multivariate LSTM model.
# You also need to perform h- step prediction using LSTM model.
# You can use tensorflow package in python for this section.

energy = pd.read_csv("energy_demand.csv")
energy['Time_stamp'] = pd.to_datetime(energy['Time_stamp'], format="%Y-%m-%d %H:%M")
energy.sort_values(by='Time_stamp', inplace=True)
energy.index = energy["Time_stamp"]
energy = energy.iloc[5:5005, 1:5]

X = energy[['kWh', 'kVAR', 'kVARh']].values
y = energy[['kW']].values

# define h for h-step prediction
h = 5

# split the data into train and test sets
train_size = int(len(X) * 0.8)
train, test = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# define the input and output sequences
train_X, train_y = train[:-h], train_y[h:]
test_X, test_y = test[:-h], test_y[h:]

# reshape the input to 3D format (samples, timesteps, features)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(train_y.shape[1]))
model.compile(loss='mae', optimizer='adam')

# fit the model
model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# make predictions
yhat = model.predict(test_X)

# evaluate the model using mean squared error (MSE)
mse = np.mean((yhat - test_y)**2)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse}")

# calculate Residual Mean and Residual Variance
residuals = test_y - yhat
residual_mean = residuals.mean()
residual_var = residuals.var()

print(f'Residual Mean: {residual_mean:.4f}')
print(f'Residual Variance: {residual_var:.4f}')

# perform h-step prediction
if len(test_X) >= h:
    x_input = test_X[-h:]
else:
    x_input = x_input.reshape((h, 1, 1))
    predictions = []
    for i in range(h):
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat)
        x_input = np.concatenate((x_input[1:, :, :], yhat.reshape(1, 1, 1)), axis=0)

# print the h-step predictions
print(f"{h}-step predictions: {np.array(predictions).squeeze()}")

plt.plot(energy.index[-len(yhat)-1:-1], test_y, label='True')
plt.plot(energy.index[-len(yhat)-1:-1], yhat, label='Predicted')
plt.xlabel('Date')
plt.ylabel('kW')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()