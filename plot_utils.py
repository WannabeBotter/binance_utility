import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def plot_correlation(series_x, series_y, func = None, func_label: str = '$y = %s x %s$', title: str = None, xaxis_label: str = 'x', yaxis_label: str = 'y', legend_loc: str = 'best'):

    _df = pl.DataFrame({'x': series_x, 'y': series_y}).fill_nan(None).drop_nulls()
    _corr = np.corrcoef(_df['x'], _df['y'])
    _y_std = _df['y'].std()
    _y_mean = _df['y'].mean()
    _x_std = _df['x'].std()
    _x_mean = _df['x'].mean()
    
    _std_range = 3
    _y_max = _y_mean + _std_range * _y_std
    _y_min = _y_mean - _std_range * _y_std
    _x_max = _x_mean + _std_range * _x_std
    _x_min = _x_mean - _std_range * _x_std
    
    fig, ax = plt.subplots(2, 2, sharex = 'col', sharey = 'row', gridspec_kw = {'width_ratios': [2, 0.5], 'height_ratios': [2, 0.5]}, figsize = (8, 8))
    
    # レンジごとの平均値を階段状にプロット
    _x_sections = []
    _y_means = []
    for i in range(_std_range * 4 + 1):
        __df = _df.filter((pl.col('x') >= _x_min + 0.5 * _x_std * i) & (pl.col('x') < _x_min + 0.5 * _x_std * (i + 1)))
        _x_sections.append(_x_min + 0.5 * _x_std * i)
        _y_means.append(__df['y'].mean())
        
    # 近似直線のプロット
    _ax = ax[0, 0]

    def _default_func(x, a, b):
        return a * x + b
    
    if func == None:
        func = _default_func
    
    _x_linspace = np.linspace(_x_min, _x_max, 50)
    _popt, _pcov = curve_fit(func, _df['x'], _df['y'])
    _ax.plot(_x_linspace, func(_x_linspace, *_popt), color = 'green', label = func_label % (f'{_popt[0]:.4e}',  f'{_popt[1]:+.4e}'))
    print(f"_popt = {_popt}")

    _y_pred = func(_df['x'], *_popt)
    _mse = mean_squared_error(_df['y'], _y_pred)

    # 散布図
    _ax.scatter(_df['x'], _df['y'], s = 0.1)

    # ステップ状の平均値
    _ax.step(_x_sections, _y_means, 'red', where = 'post', alpha=0.5)

    _ax.set_title(title)
    _ax.set_xlabel(xaxis_label)
    _ax.set_ylabel(yaxis_label)
    _ax.set_xlim([_x_min, _x_max])
    _ax.set_ylim([_y_min, _y_max])
    _ax.set_xticks([_x_mean, _x_mean - 2 * _x_std, _x_mean - 4 * _x_std, _x_mean + 2 * _x_std, _x_mean + 4 * _x_std])
    _ax.set_yticks([_y_mean, _y_mean - 2 * _y_std, _y_mean - 4 * _y_std, _y_mean + 2 * _y_std, _y_mean + 4 * _y_std])
    _ax.grid(axis = 'both')
    _ax.axvline(0, color = 'red', linestyle = 'dotted', linewidth = 1)
    _ax.axhline(0, color = 'red', linestyle = 'dotted', linewidth = 1)
    _ax.text(0.01, 0.99, f'x vs y CORR = {_corr[0][1]:0.4f}, y vs y_pred MSE = {_mse:0.4f}', va = 'top', ha = 'left', transform = _ax.transAxes)
    _ax.legend(loc = legend_loc)

    # ヒストグラム
    _ax = ax[1, 0]
    _ax.hist(_df['x'], bins = 50, range = [_x_min, _x_max])
    _ax.grid(axis = 'both')
    _ax.axvline(0, color='red', linestyle = 'dotted', linewidth = 1)
    
    _ax = ax[0, 1]
    _ax.hist(_df['y'], bins = 50, orientation = 'horizontal', range = [_y_min, _y_max])
    _ax.grid(axis = 'both')
    _ax.axhline(0, color = 'red', linestyle = 'dotted', linewidth = 1)
    
    ax[1, 1].remove()
