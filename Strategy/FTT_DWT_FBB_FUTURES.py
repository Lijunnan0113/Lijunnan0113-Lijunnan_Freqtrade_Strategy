import numpy as np
import scipy.fft
from scipy.fft import rfft, irfft
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow

from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

# Get rid of pandas warnings during backtesting
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import custom_indicators as cta

import pywt
import scipy


class FTT_DWT_FBB_FUTURES(IStrategy):

    INTERFACE_VERSION = 3

    levarage_input = 3.0

    # Do *not* hyperopt for the roi and stoploss spaces

    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    timeframe = '5m'
    inf_timeframe = '15m'

    use_custom_stoploss = True

    # Recommended
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Required
    startup_candle_count: int = 400 # must be power of 2

    process_only_new_candles = True

    trading_mode = "futures"
    margin_mode = "isolated"
    can_short = True

    custom_trade_info = {}

    ###################################

    # Strategy Specific Variable Storage

    ## Hyperopt Variables
    
    # FBB_ hyperparams
    buy_bb_gain = DecimalParameter(0.01, 0.50, decimals=2, default=0.03, space='buy', load=True, optimize=True)
    buy_fisher_wr = DecimalParameter(-0.99, -0.75, decimals=2, default=-0.5, space='buy', load=True, optimize=True)
    buy_force_fisher_wr = DecimalParameter(-0.99, -0.85, decimals=2, default=-0.99, space='buy', load=True, optimize=True)

    sell_bb_gain = DecimalParameter(0.7, 1.5, decimals=2, default=0.8, space='sell', load=True, optimize=True)
    sell_fisher_wr = DecimalParameter(0.75, 0.99, decimals=2, default=0.75, space='sell', load=True, optimize=True)
    sell_force_fisher_wr = DecimalParameter(0.85, 0.99, decimals=2, default=0.99, space='sell', load=True, optimize=True)

    # FFT  hyperparams
    entry_fft_diff = DecimalParameter(0.0, 5.0, decimals=1, default=2.0, space='buy', load=True, optimize=True)
    entry_fft_dev = DecimalParameter(-4.0, 0.00, decimals=1, default=-0.1, space='buy', load=True, optimize=True)
        # buy_fft_cutoff = DecimalParameter(1/16.0, 1/3.0, decimals=2, default=1/5.0, space='buy', load=True, optimize=True)

    exit_fft_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-0.01, space='buy', load=True, optimize=True)
    exit_fft_dev = DecimalParameter(0.00, 4.0, decimals=1, default=1.0, space='buy', load=True, optimize=True)


    fft_window = startup_candle_count
    fft_lookahead = 0
    
    dwt_window = startup_candle_count
    
    # DWT  hyperparams
    entry_long_dwt_diff = DecimalParameter(0.0, 5.0, decimals=1, default=2.0, space='buy', load=True, optimize=True)
    entry_short_dwt_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-2.0, space='buy', load=True, optimize=True)
    exit_long_dwt_diff = DecimalParameter(-5.0, 0.0, decimals=1, default=-2.0, space='sell', load=True, optimize=True)
    exit_short_dwt_diff = DecimalParameter(0.0, 5.0, decimals=1, default=-2.0, space='sell', load=True, optimize=True)

    entry_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'macd', 'adx'], default='rmi', space='buy',
                                            load=True, optimize=True)

    # Custom exit Profit (formerly Dynamic ROI)
    cexit_long_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    cexit_long_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_long_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_long_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_long_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_long_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_long_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_long_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_long_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_short_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)

    cexit_short_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    cexit_short_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    cexit_short_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    cexit_short_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    cexit_short_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cexit_short_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    cexit_short_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    cexit_short_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    # Custom Stoploss
    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    cstop_max_stoploss =  DecimalParameter(-0.30, -0.01, default=-0.10, space='sell', load=True, optimize=True)

    ###################################

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs
    
    ###################################

    """
    Indicator Definitions
    """
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        if self.levarage_input > max_leverage:
            return max_leverage

        return self.levarage_input 

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # Base pair informative timeframe indicators
        curr_pair = metadata['pair']
        informative = self.dp.get_pair_dataframe(pair=curr_pair, timeframe=self.inf_timeframe)

        # FFT
        informative['fft_dev'] = informative['close'].rolling(window=self.fft_window).apply(self.scaledModel)
        informative['fft_dev'] = informative['fft_dev'].fillna(0) # missing data can cause issue with ta functions
        informative['fft_slope'] = ta.LINEARREG_SLOPE(informative['fft_dev'], timeperiod=60)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        dataframe['scaled'] = dataframe['close'].rolling(window=self.fft_window).apply(self.scaledData)
        dataframe['fft_dev'] = dataframe[f"fft_dev_{self.inf_timeframe}"]
        dataframe['fft_slope'] = dataframe[f"fft_slope_{self.inf_timeframe}"]
        dataframe['fft_dev_diff'] =  (dataframe['fft_dev'] - dataframe['scaled'])

        # DWT
        informative['dwt_model'] = informative['close'].rolling(window=self.dwt_window).apply(self.model)

        # merge into normal timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        # calculate predictive indicators in shorter timeframe (not informative)

        dataframe['dwt_model'] = dataframe[f"dwt_model_{self.inf_timeframe}"]
        dataframe['dwt_model_diff'] = 100.0 * (dataframe['dwt_model'] - dataframe['close']) / dataframe['close']

        # FisherBB

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=30)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        # Williams %R
        dataframe['wr'] = 0.02 * (self.williams_r(dataframe, period=30) + 50.0)

        # Combined Fisher RSI and Williams %R
        dataframe['fisher_wr'] = (dataframe['wr'] + dataframe['fisher_rsi']) / 2.0

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Custom Stoploss

        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False

        # MA Streak: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
        # dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        # Trends

        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['candle-dn-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() <= 2, 1, 0)


        # RMI: https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/
        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)
        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() <= 2, 1, 0)

        # dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        # dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()
        #
        # dataframe['rmi-up'] = np.where(dataframe['rmi'] > dataframe['rmi'].shift(), 1, 0)
        # dataframe['rmi-up-count'] = dataframe['rmi-up'].rolling(8).sum()

        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['adx-up-trend'] = np.where(
            (
                    (dataframe['adx'] > 20.0) &
                    (dataframe['dm_plus'] > dataframe['dm_minus'])
            ), 1, 0)
        dataframe['adx-dn-trend'] = np.where(
            (
                    (dataframe['adx'] > 20.0) &
                    (dataframe['dm_plus'] < dataframe['dm_minus'])
            ), 1, 0)

        # Indicators used only for ROI and Custom Stoploss
        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

        return dataframe

    ###################################


    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def model(self, a: np.ndarray) -> float:
        #must return scalar, so just calculate prediction and take last value
        # npredict = self.buy_fft_lookahead.value
        model = self.fourierModel(np.array(a))
        length = len(model)
        return model[length-1]
    def fourierModel(self, x):

        n = len(x)
        xa = np.array(x)


        # compute the fft
        fft = scipy.fft.fft(xa, n)

        # compute power spectrum density
        # squared magnitude of each fft coefficient
        psd = fft * np.conj(fft) / n
        threshold = 20
        fft = np.where(psd<threshold, 0, fft)

        # inverse fourier transform
        ifft = scipy.fft.ifft(fft)

        ifft = ifft.real

        ldiff = len(ifft) - len(xa)
        model = ifft[ldiff:]

        return model
    def scaledModel(self, a: np.ndarray) -> float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        scaled.fillna(0, inplace=True)

        # get the Fourier model
        model = self.fourierModel(scaled)

        length = len(model)
        return model[length-1]
    def scaledData(self, a: np.ndarray) -> float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        # scaled.fillna(0, inplace=True)

        length = len(scaled)
        return scaled.ravel()[length-1]

    def dwtModel(self, data):

        # the choice of wavelet makes a big difference
        # for an overview, check out: https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        # wavelet = 'db1'
        # wavelet = 'bior1.1'bior3.3 symmetric smooth
        # wavelet = 'haar' # deals well with harsh transitions
        # wavelet = 'bior3.3'
        wavelet = 'haar'
        level = 1
        wmode = "smooth"
        length = len(data)

        coeff = pywt.wavedec(data, wavelet, mode=wmode)

        # remove higher harmonics
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(length))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

        # inverse transform
        model = pywt.waverec(coeff, wavelet, mode=wmode)

        return model

    def model(self, a: np.ndarray) -> float:
        #must return scalar, so just calculate prediction and take last value
        # model = self.dwtModel(np.array(a))

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        x_notrend = (a - w_mean) / w_std

        # get DWT model of data
        restored_sig = self.dwtModel(x_notrend)

        # re-trend
        model = (restored_sig * w_std) + w_mean

        length = len(model)
        return model[length-1]

    def scaledModel(self, a: np.ndarray) -> float:
        #must return scalar, so just calculate prediction and take last value
        # model = self.dwtModel(np.array(a))

        # de-trend the data
        w_mean = a.mean()
        w_std = a.std()
        x_notrend = (a - w_mean) / w_std

        # get DWT model of data
        model = self.dwtModel(x_notrend)

        length = len(model)
        return model[length-1]

    def scaledData(self, a: np.ndarray) -> float:

        # scale the data
        standardized = a.copy()
        w_mean = np.mean(standardized)
        w_std = np.std(standardized)
        scaled = (standardized - w_mean) / w_std
        # scaled.fillna(0, inplace=True)

        length = len(scaled)
        return scaled.to_numpy()[length-1]

    def predict(self, a: np.ndarray, npredict: int = 1) -> float:
        # Normalize the input data
        w_mean = np.mean(a)
        w_std = np.std(a)
        standardized = (a - w_mean) / w_std

        # Generate predictions using DWT model
        dwt_model = self.dwtModel(standardized)
        dwt_predicted = (dwt_model * w_std) + w_mean

        length = len(dwt_predicted)
        if npredict == 0:
            return dwt_predicted[length - 1]
        else:
            # Ensure there's enough data to perform interpolation
            if length < 2:
                raise ValueError("Not enough data to perform prediction.")
            
            # Use cubic spline interpolation for future predictions
            x = np.arange(length)
            f = scipy.interpolate.UnivariateSpline(x, dwt_predicted, k=5, s=0)
            prediction = f(length - 1 + npredict)

        return prediction

    # Williams %R
    def williams_r(self, dataframe: DataFrame, period: int = 30) -> Series:
        """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
            of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
            Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
            of its recent trading range.
            The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = dataframe["high"].rolling(center=False, window=period).max()
        lowest_low = dataframe["low"].rolling(center=False, window=period).min()

        WR = Series(
            (highest_high - dataframe["close"]) / (highest_high - lowest_low),
            name=f"{period} Williams %R",
        )

        return WR * -100

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        current_profit = trade.calc_profit_ratio(current_candle['close'])

        if trade.is_short:
            return self.custom_exit_short(pair, trade, current_time, current_rate, current_profit)
        else:
            return self.custom_exit_long(pair, trade, current_time, current_rate, current_profit)

    ###################################

    """
    entry Signal
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # short_conditions = []
        # long_conditions = []
        # dataframe.loc[:, 'enter_tag'] = ''
        # dataframe.loc[:, 'long'] = ''  # 初始化 buy 列为 0，表示没有买入
        # dataframe.loc[:, 'short'] = ''

        # checks for long/short conditions
        if (self.entry_trend_type.value != 'rmi'):
            long_cond = (dataframe['rmi-dn-trend'] == 1)
            short_cond = (dataframe['rmi-up-trend'] == 1)
        elif (self.entry_trend_type.value != 'ssl'):
            long_cond = (dataframe['ssl-dir'] == 'down')
            short_cond = (dataframe['ssl-dir'] == 'up')
        elif (self.entry_trend_type.value != 'candle'):
            long_cond = (dataframe['candle-dn-trend'] == 1)
            short_cond = (dataframe['candle-up-trend'] == 1)
        elif (self.entry_trend_type.value != 'macd'):
            long_cond = (dataframe['macdhist'] < 0.0)
            short_cond = (dataframe['macdhist'] > 0.0)
        elif (self.entry_trend_type.value != 'adx'):
            long_cond = (dataframe['adx-dn-trend'] == 1)
            short_cond = (dataframe['adx-up-trend'] == 1)

            # long_conditions.append(long_cond)
            # short_conditions.append(short_cond)

        # === FFT 用于识别长期趋势 ===
        fft_long_trend_cond = (
            (dataframe['fft_slope'] >= 0.0) &  # FFT 斜率为正，表示上升趋势
            (dataframe['fft_dev'] < self.entry_fft_dev.value) & # 偏差小于阈值
            (qtpylib.crossed_above(dataframe['fft_dev_diff'], self.entry_fft_diff.value))  # 偏差差异突破阈值
        )
        # long_conditions.append(fft_long_trend_cond)
        
        fft_short_trend_cond = (
            (dataframe['fft_slope'] <= 0.0) &  # FFT 斜率为负，表示下降趋势
            (dataframe['fft_dev'] > self.exit_fft_dev.value) & # 偏差大于阈值
            (qtpylib.crossed_below(dataframe['fft_dev_diff'], self.exit_fft_diff.value))  # 偏差差异突破阈值
        )
        # short_conditions.append(fft_short_trend_cond)

        # === DWT 用于检测短期波动 ===
        dwt_long_cond = (
            qtpylib.crossed_above(dataframe['dwt_model_diff'], self.entry_long_dwt_diff.value)
        )
        long_spike_cond = (
                dataframe['dwt_model_diff'] < 2.0 * self.entry_long_dwt_diff.value
        )


        dwt_short_cond = (
            qtpylib.crossed_below(dataframe['dwt_model_diff'], self.entry_short_dwt_diff.value)
        )
        short_spike_cond = (
                dataframe['dwt_model_diff'] > 2.0 * self.entry_short_dwt_diff.value
        )

        # FFT 和 DWT 条件都满足
        fft_dwt_long = [fft_long_trend_cond , dwt_long_cond]
        fft_dwt_short = [fft_short_trend_cond , dwt_short_cond]
        dataframe.loc[reduce(lambda x, y: x & y, fft_dwt_long), ["enter_long", "enter_tag"]] = (1, "fft_dwt_long")
        dataframe.loc[reduce(lambda x, y: x & y, fft_dwt_short), ["enter_short", "enter_tag"]] = (1, "fft_dwt_short")    

        # === 设置条件标签和买入信号 ===
        dataframe.loc[dwt_long_cond & long_spike_cond & long_cond, 'enter_tag'] = 'dwt_long_cond'
        dataframe.loc[dwt_long_cond & long_spike_cond & long_cond, 'enter_long'] = 1  # 设定买入
        # === 设置条件标签和买入信号 ===
        dataframe.loc[dwt_short_cond & short_spike_cond & short_cond, 'enter_tag'] = 'dwt_short_cond'
        dataframe.loc[dwt_short_cond & short_spike_cond & short_cond, 'enter_short'] = 1  # 设定买入

        # if long_conditions:
            # dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1
        # if short_conditions:
            # dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1
        return dataframe


    ###################################

    """
    exit Signal
    """


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'exit_tag'] = ''
        
        # === DWT 用于短期波动的反转 ===
    
        # Long Processing
        long_dwt_cond = (
            qtpylib.crossed_below(dataframe['dwt_model_diff'], self.exit_long_dwt_diff.value)
        )
    
        long_spike_cond = (
            dataframe['dwt_model_diff'] > 2.0 * self.exit_long_dwt_diff.value
        )
    
        long_conditions.append(long_dwt_cond)
        long_conditions.append(long_spike_cond)
        
        dataframe.loc[long_dwt_cond, 'exit_tag'] += 'long_dwt_exit '

        
        # Short Processing
        short_dwt_cond = (
            qtpylib.crossed_above(dataframe['dwt_model_diff'], self.exit_short_dwt_diff.value)
        )

        short_spike_cond = (
            dataframe['dwt_model_diff'] < 2.0 * self.exit_short_dwt_diff.value
        )
    
        short_conditions.append(short_dwt_cond)
        short_conditions.append(short_spike_cond)
        
        dataframe.loc[short_dwt_cond, 'exit_tag'] += 'short_dwt_exit '

        # 设置多头退出信号
        if long_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'exit_long'] = 1
        # 设置空头退出信号
        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'exit_short'] = 1

        return dataframe

    ###################################

    # the custom stoploss/exit logic is adapted from Solipsis by werkkrew (https://github.com/werkkrew/freqtrade-strategies)

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        # 如果亏损达到或超过 -10%，直接止损
        # if current_profit <= -0.08:
            # return 0.01  # 止损

        # limit stoploss
        if current_profit <  self.cstop_max_stoploss.value:
            return 0.01

        # Determine how we exit when we are in a loss
        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on rate of change
                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':
                # Dynamic bailout based on time, unless time_trend is true and there is a potential reversal
                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1

    ###################################

    """
    Custom exit
    """

    def custom_exit_long(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                             current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0.0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0.0, (max_profit - self.cexit_long_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_long_roi_type.value == 'static':
            min_roi = self.cexit_long_roi_start.value
        elif self.cexit_long_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_long_roi_start.value, self.cexit_long_roi_end.value, 0,
                                       self.cexit_long_roi_time.value, trade_dur)
        elif self.cexit_long_roi_type.value == 'step':
            if trade_dur < self.cexit_long_roi_time.value:
                min_roi = self.cexit_long_roi_start.value
            else:
                min_roi = self.cexit_long_roi_end.value

        # Determine if there is a trend
        if self.cexit_long_trend_type.value == 'rmi' or self.cexit_long_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.cexit_long_trend_type.value == 'ssl' or self.cexit_long_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.cexit_long_trend_type.value == 'candle' or self.cexit_long_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        # Don't exit if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful exit message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a exit, maybe
            if self.cexit_long_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_long_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.cexit_long_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_roi'
                elif self.cexit_long_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None

    def custom_exit_short(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0.0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0.0, (max_profit - self.cexit_short_pullback_amount.value))
        in_trend = False

        # Determine our current ROI point based on the defined type
        if self.cexit_short_roi_type.value == 'static':
            min_roi = self.cexit_short_roi_start.value
        elif self.cexit_short_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.cexit_short_roi_start.value, self.cexit_short_roi_end.value, 0,
                                       self.cexit_short_roi_time.value, trade_dur)
        elif self.cexit_short_roi_type.value == 'step':
            if trade_dur < self.cexit_short_roi_time.value:
                min_roi = self.cexit_short_roi_start.value
            else:
                min_roi = self.cexit_short_roi_end.value

        # Determine if there is a trend
        if self.cexit_short_trend_type.value == 'rmi' or self.cexit_short_trend_type.value == 'any':
            if last_candle['rmi-dn-trend'] == 1:
                in_trend = True
        if self.cexit_short_trend_type.value == 'ssl' or self.cexit_short_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'down':
                in_trend = True
        if self.cexit_short_trend_type.value == 'candle' or self.cexit_short_trend_type.value == 'any':
            if last_candle['candle-dn-trend'] == 1:
                in_trend = True

        # Don't exit if we are in a trend unless the pullback threshold is met
        if in_trend == True and current_profit > 0:
            # Record that we were in a trend for this trade/pair for a more useful exit message later
            self.custom_trade_info[trade.pair]['had-trend'] = True
            # If pullback is enabled and profit has pulled back allow a exit, maybe
            if self.cexit_short_pullback.value == True and (current_profit <= pullback_value):
                if self.cexit_short_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'short_intrend_pullback_roi'
                elif self.cexit_short_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'short_intrend_pullback_roi'
                    else:
                        return 'short_intrend_pullback_noroi'
            # We are in a trend and pullback is disabled or has not happened or various criteria were not met, hold
            return None
        # If we are not in a trend, just use the roi value
        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'short_trend_roi'
                elif self.cexit_short_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'short_trend_noroi'
            elif current_profit > min_roi:
                return 'short_notrend_roi'
        else:
            return None
