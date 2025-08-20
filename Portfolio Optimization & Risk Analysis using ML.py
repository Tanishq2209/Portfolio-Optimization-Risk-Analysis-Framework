# Importing Libraries

# Data handling and statistical analysis 
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, t
import pandas_datareader.data as web

# Data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Optimization and allocation (Machine Learning)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Datetime and hiding warnings 
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os


# List of stocks used as an example for this project
tickers = ['SPY.US', 'NVDA.US', 'TSLA.US', 'AMZN.US', 'GOOG.US', 'AAPL.US']

combined_df = pd.DataFrame()
for ticker in tickers:
    file_path = os.path.join( f"{ticker}.csv")
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        close = df[['Close']].copy()
        close.columns = [ticker.replace('.US', '')]  # Rename column to ticker
        combined_df = pd.merge(combined_df, close, left_index=True, right_index=True, how='outer')
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

combined_df.sort_index(inplace=True)
combined_df.to_csv('Stock Prices data with SP500.csv')
df = pd.read_csv('Stock Prices data with SP500.csv')


# Ensure Date column is in datetime format and setting as index for normalization 
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Normalize each stock price
normalized_df = df / df.iloc[0] * 100
normalized_df = normalized_df.reset_index()


# Plotting each normalized stock
for stocks in ['NVDA', 'TSLA', 'AMZN', 'GOOG', 'AAPL']:
    sns.lineplot(data=normalized_df, x='Date', y=stocks, label=stocks)
plt.title('Normalized Stocks Performance (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Normalized Price', fontsize=12)
plt.legend(title='Stocks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()
plt.show()


# Calculate daily percentage change (simple returns)
ret_port = df.pct_change()
fig = px.line(ret_port, width=1000, height=600)
px.line(ret_port)


# Correlation matrix and Heatmap
correlation_matrix = normalized_df[['NVDA', 'TSLA', 'AMZN', 'GOOG', 'AAPL']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Normalized Stock Prices', fontsize=16)
plt.tight_layout()
plt.show()

# We generally do log return instead of return for the stationarity
log_ret = np.log(df / df.shift(1))
log_ret
# Log returns by dropping SPY
log_ret1 = log_ret.drop(columns='SPY')

# Normalized Portfolio Weights
np.random.seed(30)
weights = np.random.random((5, 1))
weights /= np.sum(weights)

exp_ret = log_ret1.mean().dot(weights.flatten()) * 252
print(f'Expected Annual Return: {exp_ret:.4f}')

exp_vol = np.sqrt(weights.T.dot(252*log_ret1.cov().dot(weights)))
print(f'Expected Volatility (Risk): {exp_vol.item():.4f}')

sr = exp_ret / exp_vol
print(f'Sharpe Ratio (r_f = 0): {sr.item():.4f}')


# Monte Carlo Simulation
n = 10000
num_assets = len(log_ret1.columns)

port_weights = np.zeros((n, num_assets))
port_return = np.zeros(n)
port_volatility = np.zeros(n)
port_sr = np.zeros(n)
port_sortino = np.zeros(n)
port_treynor = np.zeros(n)
port_m2 = np.zeros(n)

rf = 0.0  # Risk-free rate
benchmark_ret = log_ret['SPY'].mean() * 252
benchmark_vol = log_ret['SPY'].std() * np.sqrt(252)

# Simulation looping for all metrics
for i in range(n):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    port_weights[i, :] = weights

    # Expected return
    exp_ret = log_ret1.mean().dot(weights) * 252
    port_return[i] = exp_ret

    # Volatility
    exp_vol = np.sqrt(weights.T.dot(252 * log_ret1.cov().dot(weights)))
    port_volatility[i] = exp_vol

    # Sharpe Ratio
    sr = (exp_ret - rf) / exp_vol
    port_sr[i] = sr

    # Sortino Ratio (downside deviation)
    downside = log_ret1[log_ret1 < 0].std().dot(weights) * np.sqrt(252)
    sortino = (exp_ret - rf) / downside
    port_sortino[i] = sortino

    # Treynor Ratio (uses benchmark beta)
    # Covariance between each asset and SPY
    cov_matrix = log_ret.cov()
    cov_with_spy = cov_matrix.loc[log_ret1.columns, 'SPY']  # Series of covariances with shape (5, )
    # Dot with weights (flattened to match shape)
    cov_with_benchmark = cov_with_spy.dot(weights.flatten())

    beta = cov_with_benchmark / log_ret['SPY'].var()
    treynor = (exp_ret - rf) / beta
    port_treynor[i] = treynor

    # M² (Modigliani-Modigliani)
    m2 = sr * benchmark_vol + rf
    port_m2[i] = m2

ind_sr = port_sr.argmax()
ind_sortino = port_sortino.argmax()
ind_treynor = port_treynor.argmax()
ind_m2 = port_m2.argmax()

plt.scatter(port_volatility, port_return, c=port_sr, cmap='plasma', alpha=0.6)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility', fontsize=15)
plt.ylabel('Return', fontsize=15)
plt.title('Efficient Frontier with Optimal Portfolios', fontsize=16)

# Highlight optimal portfolios
plt.scatter(port_volatility[ind_sr], port_return[ind_sr], c='blue', s=50,  label='Max Sharpe')
plt.scatter(port_volatility[ind_sortino], port_return[ind_sortino], c='green', s=50, marker='+', label='Max Sortino')
plt.scatter(port_volatility[ind_treynor], port_return[ind_treynor], c='purple', s=30, marker='^', label='Max Treynor')
plt.scatter(port_volatility[ind_m2], port_return[ind_m2], c='red', s=20, marker='*', label='Max M²')
plt.legend()
plt.grid(True)
plt.show()


# Clustering-Based Portfolio Construction
def ml_enhanced_optimization(df, n_clusters=3):
    """Using the clustering method to group similar assets before optimization"""
    
    # Calculate returns and volatility features
    returns = df.pct_change().dropna()
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Create feature matrix
    features = np.column_stack([annual_returns, annual_volatility])
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', s=100)
    for i, ticker in enumerate(df.columns):
        plt.annotate(ticker, (pca_result[i, 0], pca_result[i, 1]))
    plt.colorbar(scatter, label='Cluster')
    plt.title('Asset Clustering based on Return and Volatility')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid()
    plt.show()
    return clusters
clusters = ml_enhanced_optimization(df.drop(columns=['SPY']))


# Random Forest Feature Importance for Asset Selection
def feature_importance_analysis(df):
    """Analyze which assets contribute most to portfolio performance"""
    returns = df.pct_change().dropna()
    X = returns.iloc[:-1]  # Features (previous day returns)
    y = returns.iloc[1:].mean(axis=1)  # Target (next day portfolio return)
    
    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature importance
    importance = pd.DataFrame({
        'Asset': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.bar(importance['Asset'], importance['Importance'])
    plt.title('Asset Importance for Portfolio Returns')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return importance
asset_importance = feature_importance_analysis(df)

# VaR (Value at Risk) and CVaR (Conditional Value at Risk)
def calculate_var_cvar(weights, returns, alpha=0.95, method='historical'):
    """Calculate VaR and CVaR for a portfolio"""
    portfolio_returns = returns.dot(weights)
    if method == 'historical':
        # Historical VaR/CVaR
        var = np.percentile(portfolio_returns, (1 - alpha) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
    elif method == 'parametric':
        # Parametric (Normal distribution)
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        var = norm.ppf(1 - alpha, mu, sigma)
        cvar = mu - sigma * (norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha))
        
    elif method == 't-distribution':
        # Student's t-distribution
        params = t.fit(portfolio_returns)
        var = t.ppf(1 - alpha, *params)
        # More robust CVaR calculation for t-distribution
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        if np.isnan(cvar):
            cvar = var  # Fallback if no returns below VaR
    return var, cvar

def monte_carlo_var(weights, returns, n_simulations=10000, alpha=0.95):
    """Monte Carlo simulation for VaR"""
    mu = returns.mean()
    cov = returns.cov()
    
    # Generate simulated returns
    simulated_returns = np.random.multivariate_normal(mu, cov, n_simulations)
    portfolio_simulated = simulated_returns.dot(weights)
    
    # Calculate VaR and CVaR
    var = np.percentile(portfolio_simulated, (1 - alpha) * 100)
    cvar = portfolio_simulated[portfolio_simulated <= var].mean()
    return var, cvar

def enhanced_monte_carlo_simulation(df, n_simulations=10000, alpha=0.95):
    """Enhanced simulation with risk measures"""
    returns = df.pct_change().dropna()
    portfolio_assets = [col for col in df.columns if col != 'SPY']
    returns_portfolio = returns[portfolio_assets]
    results = {}
    
    # Creating separate columns for each asset's weight
    for j, asset in enumerate(portfolio_assets):
        results[f'weight_{asset}'] = np.zeros(n_simulations)
    
    # Add other metrics
    metrics = ['return', 'volatility', 'sharpe', 
               'var_historical', 'cvar_historical', 
               'var_param', 'cvar_param', 'var_mc', 'cvar_mc']
    
    for metric in metrics:
        results[metric] = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        weights = np.random.random(len(portfolio_assets))
        weights /= weights.sum()
        
        # Storing weights in separate columns
        for j, asset in enumerate(portfolio_assets):
            results[f'weight_{asset}'][i] = weights[j]
        
        # Basic metrics
        portfolio_ret = returns_portfolio.dot(weights)
        exp_ret = portfolio_ret.mean() * 252
        exp_vol = np.sqrt(weights.T.dot(returns_portfolio.cov().dot(weights)) * 252)
        
        results['return'][i] = exp_ret
        results['volatility'][i] = exp_vol
        
        if exp_vol > 0:
            results['sharpe'][i] = (exp_ret - 0.02) / exp_vol
        else:
            results['sharpe'][i] = np.nan
        
        # Risk measures with error handling
        try:
            var_hist, cvar_hist = calculate_var_cvar(weights, returns_portfolio, alpha, 'historical')
            results['var_historical'][i] = var_hist
            results['cvar_historical'][i] = cvar_hist
        except:
            results['var_historical'][i] = np.nan
            results['cvar_historical'][i] = np.nan
        
        try:
            var_param, cvar_param = calculate_var_cvar(weights, returns_portfolio, alpha, 'parametric')
            results['var_param'][i] = var_param
            results['cvar_param'][i] = cvar_param
        except:
            results['var_param'][i] = np.nan
            results['cvar_param'][i] = np.nan
        
        try:
            var_mc, cvar_mc = monte_carlo_var(weights, returns_portfolio, 5000, alpha)
            results['var_mc'][i] = var_mc
            results['cvar_mc'][i] = cvar_mc
        except:
            results['var_mc'][i] = np.nan
            results['cvar_mc'][i] = np.nan
    
    return pd.DataFrame(results)

# Running the enhanced simulation
enhanced_results = enhanced_monte_carlo_simulation(df, n_simulations=5000)
print("Simulation completed successfully!")
print(f"Shape of results: {enhanced_results.shape}")
print(f"Columns: {enhanced_results.columns.tolist()}")
print(f"First few rows:")
print(enhanced_results.head())

# Sensitivity Analysis Framework
def sensitivity_analysis(df, risk_free_rates=[0.01, 0.02, 0.03, 0.04], 
                        confidence_levels=[0.90, 0.95, 0.99],
                        market_scenarios=[-0.2, -0.1, 0.0, 0.1, 0.2]):
    """Comprehensive sensitivity analysis for the portfolio returns"""
    
    results = {}
    returns = df.pct_change().dropna()
    portfolio_assets = [col for col in df.columns if col != 'SPY']
    returns_portfolio = returns[portfolio_assets]
    
    # 1. Risk-free rate sensitivity
    print("RISK-FREE RATE SENSITIVITY: ")
    for rf in risk_free_rates:
        # Run optimization with different risk-free rates
        optimal_weights = optimize_portfolio(returns_portfolio, risk_free_rate=rf)
        portfolio_perf = evaluate_portfolio(optimal_weights, returns_portfolio, rf)
        results[f'rf_{rf}'] = portfolio_perf
        print(f"RF={rf}: Sharpe={portfolio_perf['sharpe']:.3f}")
    
    # 2. Confidence level sensitivity for VaR/CVaR
    print("CONFIDENCE LEVEL SENSITIVITY: ")
    optimal_weights = optimize_portfolio(returns_portfolio)
    for alpha in confidence_levels:
        var, cvar = calculate_var_cvar(optimal_weights, returns_portfolio, alpha)
        results[f'alpha_{alpha}'] = {'var': var, 'cvar': cvar}
        print(f"Alpha={alpha}: VaR={var:.4f}, CVaR={cvar:.4f}")
    
    # 3. Market scenario analysis
    print("MARKET SCENARIO ANALYSIS: ")
    for scenario in market_scenarios:
        # Adjust returns based on scenario
        adjusted_returns = returns_portfolio * (1 + scenario)
        optimal_weights = optimize_portfolio(adjusted_returns)
        portfolio_perf = evaluate_portfolio(optimal_weights, adjusted_returns)
        results[f'scenario_{scenario}'] = portfolio_perf
        print(f"Scenario={scenario}: Return={portfolio_perf['return']:.4f}")
    
    return results

def optimize_portfolio(returns, risk_free_rate=0.02):
    """Basic portfolio optimization"""
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    # Running the Monte Carlo simulation
    n_simulations = 10000
    best_sharpe = -np.inf
    best_weights = None
    for _ in range(n_simulations):
        weights = np.random.random(len(mu))
        weights /= weights.sum()
        portfolio_return = weights.dot(mu)
        portfolio_vol = np.sqrt(weights.T.dot(cov).dot(weights))
        
        if portfolio_vol > 0:
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights
    return best_weights

def evaluate_portfolio(weights, returns, risk_free_rate=0.02):
    """Evaluate portfolio performance"""
    portfolio_returns = returns.dot(weights)
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else np.nan
    return {
        'return': annual_return,
        'volatility': annual_vol,
        'sharpe': sharpe,
        'weights': weights
    }
sensitivity_results = sensitivity_analysis(df)