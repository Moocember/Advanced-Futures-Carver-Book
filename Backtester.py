import numpy as np
import pandas as pd
import scipy.stats as sps

def equityLine(roc):
    return(np.cumprod(1+roc,axis=0))

def sharpeRatio(rets,rf=0,freq=252):
    return ((np.nanmean(rets,axis=0)-((1+rf)**(1/freq)-1))/np.nanstd(rets,axis=0))*np.sqrt(freq)

def CAGR(rets,freq=252):
    output = (equityLine(rets)[-1]/equityLine(rets)[0])**(freq/len(rets))-1
    return np.nan_to_num(output)

def rolling_CAGR(prices, freq=1):
    """
    Create returns of a defined frequency from pandas dfs.
    :param prices: Asset prices.
    :param freq: Frequency of returns.
    :return: Asset returns df/series.
    """
    return ((prices / prices.shift(freq))**(252/freq) - 1).fillna(0)

def DD(rets):
    eq = equityLine(rets)
    if rets.ndim != 1:
        for i in range(len(rets.T)):
            eq[:,i] = eq[:,i] / np.maximum.accumulate(eq[:,i]) - 1
        return eq
    else:
        return eq / np.maximum.accumulate(eq) - 1

def maxDD(rets,freq=252):
    eq = equityLine(rets)
    if rets.ndim != 1:
        for i in range(len(rets.T)):
            eq[:,i] = eq[:,i] / np.maximum.accumulate(eq[:,i]) - 1
        return(np.min(eq,axis = 0))
    else:
        return(np.min(eq / np.maximum.accumulate(eq) - 1))

def calmarRatio(rets,freq=252):
    return(-1*CAGR(rets,freq)/maxDD(rets))

def sortinoRatio(rets,rf=0,freq=252):
    if rets.ndim != 1:
        output = np.zeros((len(rets.T)))
        for i in range(len(rets.T)):
            rets_idx = rets[:,i]
            output[i] = sortinoRatio(rets_idx,rf=rf,freq=freq)
        return output
    else:
        return (np.mean(rets)-rf)/np.std(rets[rets<0])*np.sqrt(freq)

def downside_vol(rets,freq=252):
    if rets.ndim != 1:
        output = np.zeros((len(rets.T)))
        for i in range(len(rets.T)):
            rets_idx = rets[:,i]
            output[i] = sortinoRatio(rets_idx,freq=freq)
        return output
    else:
        return np.std(rets[rets<0])*np.sqrt(freq)

import scipy.stats as ss
def deflated_sharpe(rets,rf=0,freq = 252):
    rets = pd.DataFrame(rets)
    base_sharpe = (rets.mean()-((1+rf)**(1/freq)-1))/rets.std()
    base_skewness = rets.skew()
    base_kurtosis = rets.kurtosis()
    dsr = ss.norm.cdf((base_sharpe*(len(rets)-1)**.5)/(1-base_skewness*base_sharpe+(base_kurtosis-1)/4*base_sharpe**2)**.5)
    if len(dsr) != 1:
        return ((dsr * base_sharpe.values)*freq**.5)
    return ((dsr * base_sharpe.values) * freq ** .5)[0]

def kelly_leverage(rets, rf=0, freq=252):
    return (np.mean(rets,axis=0) - ((1+rf)**(1/freq)-1))/(np.mean(rets,axis=0)**2 + np.var(rets, axis=0))

def deflated_kelly(rets):
    rets = pd.DataFrame(rets)

    sampleMean = rets.mean()
    sampleStdev = rets.std()
    sampleSkew = rets.skew()
    # Kelly Criterion estimate
    kelly = sampleMean / ((sampleMean ** 2) + (sampleStdev ** 2))
    mask = (sampleMean ** 2 + sampleStdev ** 2).values > (4 * sampleSkew).values
    kelly.loc[mask] = (sampleMean ** 2 + sampleStdev ** 2 - np.sqrt((sampleMean ** 2 + sampleStdev ** 2) ** 2 - 4 * sampleSkew * sampleMean)) / (2 * sampleSkew)

    if len(kelly) != 1:
        return kelly.values
    return kelly.values[0]

def calculate_kelly_probability_tuples(rets):
    # Retrieve Insight rets
    rets = pd.DataFrame(rets)
    n = len(rets)

    sampleMean = rets.mean()
    sampleStdev = rets.std()
    sampleSkew = rets.skew()
    # Kelly Criterion estimate
    kelly = deflated_kelly(rets)
    # mask = (sampleMean**2+sampleStdev**2).values > ((sampleMean ** 2 + sampleStdev ** 2) ** 2 - 4 * sampleSkew * sampleMean).values
    # kelly.loc[mask] = (sampleMean**2+sampleStdev**2-np.sqrt((sampleMean**2+sampleStdev**2)**2-4*sampleSkew*sampleMean))/(2*sampleSkew)
    # Kelly Criterion estimate standard deviation
    a = 1 / (sampleStdev ** 2)
    b = (2 * sampleMean ** 2) / (sampleStdev ** 4)
    kellyDev = np.sqrt((a + b) / (n - 1))
    if len(kellyDev) != 1:
        return np.round(1 - sps.norm.cdf(kelly / kellyDev),6)
    return np.round(1 - sps.norm.cdf(kelly / kellyDev), 6)[0]

def vol(rets,freq=252):
    return np.std(rets,axis=0)*np.sqrt(freq)

def riskAdjCAGR(rets,freq=252):
    return CAGR(rets)*(1-vol(rets,freq=freq))

def WoverL(rets):
    if rets.ndim != 1:
        output = np.zeros((len(rets.T)))
        for i in range(len(rets.T)):
            rets_idx = rets[:,i]
            output[i] = WoverL(rets_idx)
        return output
    else:
        return np.mean(rets[rets>0],axis=0)/-np.mean(rets[rets<0],axis=0)

def win_percent(rets):
    return np.sum(rets > 0, axis=0) /(np.sum(rets > 0, axis=0)+np.sum(rets < 0, axis=0))

def kelly_crit(rets):
    B = WoverL(rets)
    P = win_percent(rets)
    return (B*P-(1-P))/B

from scipy.optimize import minimize

def neg_CAGR(leverage, rets):
    return -CAGR(rets * leverage)

def solve_kelly_lev(rets):
    start_weights = [0]

    ## Constraints - positive weights, adding to 1.0
    bounds = [(0.0, 100000.0)]
    # cdict = [{'type': 'eq', 'fun': addem}]

    ans = minimize(neg_CAGR, start_weights, (rets), method='SLSQP', bounds=bounds, constraints=[], tol=0.000001)

    return ans['x'][0]

def skewer(rets):
    rets = pd.DataFrame(rets)

    sampleSkew = rets.skew().values
    return np.asarray([sampleSkew]).reshape((len(sampleSkew)))

def average_days_in_drawdown(rets,thresh=.01,freq=252):
    eql = equityLine(rets)
    mask = eql.expanding(0).max() > eql + thresh
    return mask.sum()/len(mask)




def pd_fullBacktest(rets,freq=252):
    rets_np = rets.values.astype(np.float64)
    bt = fullBacktest(rets_np,freq=freq)
    bt.columns = rets.columns
    return bt

def series_fullBacktest(rets,freq=252):
    rets_np = rets.values.astype(np.float64)
    bt = fullBacktest(rets_np, freq=freq)
    bt.index = rets.columns
    return bt

def veryFullBacktest(rets):
    idx = ["Sharpe Ratio", 'DSR', 'CAGR', 'Max DD', 'Mar Ratio', 'Sortino Ratio', 'Vol', 'Skew', 'W/L','Win Rate', 'Kelly Crit', 'Kelly Leverage', 'Deflated KL', 'Prob of Loss']
    s = sharpeRatio(rets)
    dsr = deflated_sharpe(rets)
    c = CAGR(rets)
    m = maxDD(rets)
    cr = calmarRatio(rets)
    sr = sortinoRatio(rets)
    v = vol(rets)
    skewie = sps.skew(rets,axis=0)
    WL = WoverL(rets)
    WP = win_percent(rets)
    kc = kelly_crit(rets)
    kl = kelly_leverage(rets)
    dkl = deflated_kelly(rets)
    prob = calculate_kelly_probability_tuples(rets)
    return pd.DataFrame([s, dsr, c, m, cr, sr, v, skewie, WL, WP, kc, kl, dkl, prob], index=idx)

def fullBacktest(rets,freq=252,veryfull=False):
    if isinstance(rets, np.ndarray):
        if veryfull:
            return veryFullBacktest(rets)
        else:
            idx = ["Sharpe Ratio",'CAGR','Max DD','Mar Ratio','Sortino Ratio','Vol','Skew']
            s = sharpeRatio(rets,freq=freq)
            c = CAGR(rets,freq=freq)
            m = maxDD(rets)
            cr = calmarRatio(rets,freq=freq)
            sr = sortinoRatio(rets, freq=freq)
            v = vol(rets,freq=freq)
            skewie = sps.skew(rets,axis=0)
            return pd.DataFrame([s,c,m,cr,sr,v,skewie], index=idx)
    elif isinstance(rets,pd.DataFrame):
        return pd_fullBacktest(rets,freq=freq)
    elif isinstance(rets,pd.Series):
        return fullBacktest(rets.values,freq=freq)
    else:
        raise Exception('rets must have type DataFrame, Series or np.array')

