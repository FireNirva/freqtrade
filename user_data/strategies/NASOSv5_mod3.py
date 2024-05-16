# --- Do not remove these libs ---
# --- Do not remove these libs ---
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import logging
import pandas as pd


logger = logging.getLogger(__name__)

# @Rallipanos
# @pluxury
# with help from @stash86 and @Perkmeister



def EWO(dataframe, ema_length=5, ema2_length=50):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 120
    return emadif


class NASOSv5_mod3(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 20,
        "ewo_high": 4.299,
        "ewo_high_2": 8.492,
        "ewo_low": -8.476,
        "low_offset": 0.984,
        "low_offset_2": 0.901,
        "lookback_candles": 7,
        "profit_threshold": 1.036,
        "rsi_buy": 80,
        "rsi_fast_buy": 27,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset": 1.01,
        "high_offset_2": 1.142,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.4
    }

    # Stoploss:
    stoploss = -0.3  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.001  # value loaded from strategy
    trailing_stop_positive_offset = 0.03  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy



    # SMAOffset
    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 36, default=buy_params['lookback_candles'], space='buy', optimize=True)

    profit_threshold = DecimalParameter(0.99, 1.05,
                                        default=buy_params['profit_threshold'], space='buy', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)

    rsi_buy = IntParameter(10, 80, default=buy_params['rsi_buy'], space='buy', optimize=True)
    rsi_fast_buy = IntParameter(
        10, 50, default=buy_params['rsi_fast_buy'], space='buy', optimize=True)


    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_15m = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
        'subplots': {
            'rsi': {
                'rsi': {'color': 'orange'},
                'rsi_fast': {'color': 'red'},
                'rsi_slow': {'color': 'green'},
            },
            'ewo': {
                'EWO': {'color': 'orange'}
            },
        }
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    protections = [
        # 	{
        # 		"method": "StoplossGuard",
        # 		"lookback_period_candles": 12,
        # 		"trade_limit": 1,
        # 		"stop_duration_candles": 6,
        # 		"only_per_pair": True
        # 	},
        # 	{
        # 		"method": "StoplossGuard",
        # 		"lookback_period_candles": 12,
        # 		"trade_limit": 2,
        # 		"stop_duration_candles": 6,
        # 		"only_per_pair": False
        # 	},
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.2
        },
    ]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.1):
            return 0.03
        elif (current_profit > 0.06):
            return 0.02
        elif (current_profit > 0.04):
            return 0.01
        elif (current_profit > 0.025):
            return 0.005
        elif (current_profit > 0.018):
            return 0.005

        return 0.15

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_15m

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算标准时间框架的各种技术指标，并将其添加到dataframe中。

        参数:
        dataframe (DataFrame): 包含价格数据的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了技术指标的数据框。
        """

        # 计算所有的买入移动平均线（ma_buy）值
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # 计算所有的卖出移动平均线（ma_sell）值
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # 计算50周期的赫尔移动平均线（HMA）
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # 计算100周期的指数移动平均线（EMA）
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        # 计算9周期的简单移动平均线（SMA）
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        # 计算Elliot Wave Oscillator（EWO）
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # 计算相对强弱指数（RSI）
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=20)

        # 计算快速相对强弱指数（Fast RSI）
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=5)

        # 计算慢速相对强弱指数（Slow RSI）
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=25)

        """
        ANTIPUMP测试：
        用于检测市场中的极端价格变动（Pump/Dump）。
        """

        # 计算8周期的百分比价格变化
        dataframe['pct_change'] = dataframe['close'].pct_change(periods=8)

        # 标记价格变化超过15%的情况
        dataframe['pct_change_int'] = (
                    (dataframe['pct_change'] > 0.15).astype(int) | (dataframe['pct_change'] < -0.15).astype(int))

        # 计算8周期的短期百分比价格变化
        dataframe['pct_change_short'] = dataframe['close'].pct_change(periods=8)

        # 标记短期价格变化超过8%的情况
        dataframe['pct_change_int_short'] = (
                    (dataframe['pct_change_short'] > 0.08).astype(int) | (dataframe['pct_change_short'] < -0.08).astype(
                int))

        # 检测是否存在价格快速上涨（Pump）
        dataframe['ispumping'] = (
            (dataframe['pct_change_int'].rolling(20).sum() >= 0.4)
        ).astype('int')

        # 检测是否存在长时间的价格上涨
        dataframe['islongpumping'] = (
            (dataframe['pct_change_int'].rolling(30).sum() >= 0.48)
        ).astype('int')

        # 检测是否存在短期价格快速上涨
        dataframe['isshortpumping'] = (
            (dataframe['pct_change_int_short'].rolling(10).sum() >= 0.10)
        ).astype('int')

        # 检测近期是否存在价格快速上涨
        dataframe['recentispumping'] = (dataframe['ispumping'].rolling(300).max() > 0) | (
                    dataframe['islongpumping'].rolling(
                        300).max() > 0)  # | (dataframe['isshortpumping'].rolling(300).max() > 0)

        """
        ANTIPUMP结束
        """

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        为数据框添加各种技术指标，包括不同时间框架的指标。

        参数:
        dataframe (DataFrame): 包含价格数据的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了技术指标的数据框。
        """

        # 获取15分钟时间框架的指标数据
        informative_15m = self.informative_15m_indicators(dataframe, metadata)

        # 将15分钟的指标数据合并到当前时间框架的数据框中，并向前填充缺失值
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        # 获取正常（5分钟）时间框架的指标数据
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据特定的技术指标和条件为数据框添加买入信号。

        参数:
        dataframe (DataFrame): 包含价格数据和技术指标的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了买入信号的数据框。
        """

        # 定义不应买入的条件列表
        dont_buy_conditions = []

        # 不买入条件：如果没有3%的利润空间，则不买入
        dont_buy_conditions.append(
            (
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        # 设置买入信号的条件1：EWO高、RSI快速低、价格低于买入移动平均线且成交量大于0
        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                    (dataframe['close'] < (
                                dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                    (dataframe['EWO'] > self.ewo_high.value) &
                    (dataframe['rsi'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['rsi'] < 44)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        # 设置买入信号的条件2：EWO非常高、RSI快速低、价格低于买入移动平均线且成交量大于0
        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                    (dataframe['close'] < (
                                dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                    (dataframe['EWO'] > self.ewo_high_2.value) &
                    (dataframe['rsi'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        # 设置买入信号的条件3：EWO低、RSI快速低、价格低于买入移动平均线且成交量大于0
        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                    (dataframe['close'] < (
                                dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                    (dataframe['EWO'] < self.ewo_low.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        # 应用所有不买入条件
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据特定的技术指标和条件为数据框添加卖出信号。

        参数:
        dataframe (DataFrame): 包含价格数据和技术指标的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了卖出信号的数据框。
        """

        # 定义卖出条件列表
        conditions = []

        # 卖出条件1：当前收盘价大于9周期SMA，且大于某个卖出移动平均线的高偏移值，同时RSI大于50，成交量大于0，且快速RSI大于慢速RSI
        conditions.append(
            (
                    (dataframe['close'] > dataframe['sma_9']) &
                    (dataframe['close'] > (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                    (dataframe['rsi'] > 50) &
                    (dataframe['volume'] > 0) &
                    (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
            |
            # 卖出条件2：当前收盘价小于50周期HMA，且大于某个卖出移动平均线的高偏移值，同时成交量大于0，且快速RSI大于慢速RSI
            (
                    (dataframe['close'] < dataframe['hma_50']) &
                    (dataframe['close'] > (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['volume'] > 0) &
                    (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
        )

        # 应用所有卖出条件
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe


class NASOSv5HO(NASOSv5_mod3):
    """
    NASOSv5HO 是 NASOSv5_mod3 类的子类，定义了具体的买入和卖出超空间参数，
    以及与止损和追踪止损相关的参数。
    """

    # 买入超空间参数
    buy_params = {
        "base_nb_candles_buy": 8,  # 买入基准蜡烛数量
        "ewo_high": 4.13,  # EWO高阈值
        "ewo_high_2": 4.477,  # EWO高阈值2
        "ewo_low": -19.076,  # EWO低阈值
        "lookback_candles": 27,  # 回看蜡烛数量
        "low_offset": 0.988,  # 低偏移量
        "low_offset_2": 0.974,  # 低偏移量2
        "profit_threshold": 1.049,  # 盈利阈值
        "rsi_buy": 72,  # 买入RSI阈值
        "rsi_fast_buy": 40,  # 快速买入RSI阈值
    }

    # 卖出超空间参数
    sell_params = {
        "base_nb_candles_sell": 8,  # 卖出基准蜡烛数量
        "high_offset": 1.012,  # 高偏移量
        "high_offset_2": 1.431,  # 高偏移量2
    }

    # ROI 表：从策略加载的值
    minimal_roi = {
        "0": 0.1  # 最小投资回报率
    }

    # 止损参数：从策略加载的值
    stoploss = -0.1

    # 追踪止损参数：从策略加载的值
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True


class NASOSv5PD(NASOSv5_mod3):
    """
    NASOSv5PD 是 NASOSv5_mod3 类的子类，定义了特定的买入趋势填充逻辑，
    包括额外的保护条件，例如避免在市场快速上涨（pumping）期间买入。
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据特定的技术指标和条件为数据框添加买入信号。

        参数:
        dataframe (DataFrame): 包含价格数据和技术指标的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了买入信号的数据框。
        """

        # 定义不应买入的条件列表
        dont_buy_conditions = []

        # 不买入条件1：如果没有3%的利润空间，则不买入
        dont_buy_conditions.append(
            (
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        # 不买入条件2：如果市场近期存在快速上涨（pumping），则不买入
        dont_buy_conditions.append(
            (dataframe['recentispumping'] == True)
        )

        # 设置买入信号的条件1：EWO高、RSI快速低、价格低于买入移动平均线且成交量大于0
        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                    (dataframe['close'] < (
                                dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                    (dataframe['EWO'] > self.ewo_high.value) &
                    (dataframe['rsi'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        # 设置买入信号的条件2：EWO非常高、RSI快速低、价格低于买入移动平均线且成交量大于0
        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                    (dataframe['close'] < (
                                dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                    (dataframe['EWO'] > self.ewo_high_2.value) &
                    (dataframe['rsi'] < self.rsi_buy.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                    (dataframe['rsi'] < 35)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        # 设置买入信号的条件3：EWO低、RSI快速低、价格低于买入移动平均线且成交量大于0
        dataframe.loc[
            (
                    (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                    (dataframe['close'] < (
                                dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                    (dataframe['EWO'] < self.ewo_low.value) &
                    (dataframe['volume'] > 0) &
                    (dataframe['close'] < (
                                dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        # 应用所有不买入条件
        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe


class NASOSv5SL(NASOSv5_mod3):
    """
    NASOSv5SL 是 NASOSv5_mod3 类的子类，定义了特定的卖出参数和自定义止损逻辑，
    包括硬止损、利润阈值和自定义追踪止损。
    """

    # 卖出超空间参数
    sell_params = {
        "pHSL": -0.178,  # 硬止损利润
        "pPF_1": 0.019,  # 利润阈值1，触发点，使用SL_1
        "pPF_2": 0.065,  # 利润阈值2，使用SL_2
        "pSL_1": 0.019,  # 止损1
        "pSL_2": 0.062,  # 止损2
        "base_nb_candles_sell": 12,  # 卖出基准蜡烛数量
        "high_offset": 1.01,  # 高偏移量
        "high_offset_2": 1.142,  # 高偏移量2
    }

    # 硬止损利润
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # 利润阈值1，触发点，使用SL_1
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # 利润阈值2，使用SL_2
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    trailing_stop = False  # 是否启用追踪止损
    use_custom_stoploss = True  # 是否使用自定义止损

    ## 自定义追踪止损（感谢 Perkmeister 提供的这个自定义止损，帮助策略在绿色蜡烛上行时继续持仓）
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑，根据利润阈值和当前利润计算动态止损。

        参数:
        pair (str): 交易对。
        trade (Trade): 交易对象。
        current_time (datetime): 当前时间。
        current_rate (float): 当前价格。
        current_profit (float): 当前利润。
        kwargs: 其他参数。

        返回:
        float: 计算得出的止损值。
        """

        # 硬止损利润
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # 对于利润在PF_1和PF_2之间的情况，使用线性插值计算止损值
        # 对于超过PL_2的所有利润，止损值随当前利润线性上升，对于低于PF_1的利润，使用硬止损利润
        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # 仅用于hyperopt无效返回
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)


class TrailingBuyStrat(NASOSv5_mod3):
    """
    TrailingBuyStrat 是 NASOSv5_mod3 类的子类，定义了追踪买入策略。
    包括启用追踪买入、追踪买入偏移量以及处理新蜡烛的选项。
    """

    # 如果 process_only_new_candles = True，那么你需要使用 1 分钟时间框架（并将正常策略时间框架作为参考）

    trailing_buy_order_enabled = True  # 是否启用追踪买入
    trailing_buy_offset = 0.005  # 追踪买入偏移量
    process_only_new_candles = True  # 仅处理新的蜡烛

    custom_info = dict()  # custom_info 应该是一个字典

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """
        自定义卖出逻辑。调用父类的自定义卖出方法，并重置追踪买入的状态。

        参数:
        pair (str): 交易对。
        trade (Trade): 交易对象。
        current_time (datetime): 当前时间。
        current_rate (float): 当前价格。
        current_profit (float): 当前利润。
        kwargs: 其他参数。

        返回:
        str: 卖出标签。
        """
        tag = super(TrailingBuyStrat, self).custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.custom_info[pair]['trailing_buy'] = {
                'trailing_buy_order_started': False,
                'trailing_buy_order_uplimit': 0,
                'start_trailing_price': 0,
                'buy_tag': None
            }
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        为数据框添加各种技术指标，并初始化追踪买入的信息。

        参数:
        dataframe (DataFrame): 包含价格数据和技术指标的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了技术指标的数据框。
        """
        dataframe = super(TrailingBuyStrat, self).populate_indicators(dataframe, metadata)
        if metadata["pair"] not in self.custom_info:
            self.custom_info[metadata["pair"]] = dict()
        if 'trailing_buy' not in self.custom_info[metadata['pair']]:
            self.custom_info[metadata["pair"]]['trailing_buy'] = {
                'trailing_buy_order_started': False,
                'trailing_buy_order_uplimit': 0,
                'start_trailing_price': 0,
                'buy_tag': None
            }
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        """
        确认交易退出。调用父类的确认交易退出方法，并重置追踪买入的状态。

        参数:
        pair (str): 交易对。
        trade (Trade): 交易对象。
        order_type (str): 订单类型。
        amount (float): 数量。
        rate (float): 价格。
        time_in_force (str): 有效时间。
        sell_reason (str): 卖出原因。
        kwargs: 其他参数。

        返回:
        bool: 是否确认交易退出。
        """
        val = super(TrailingBuyStrat, self).confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        self.custom_info[pair]['trailing_buy']['trailing_buy_order_started'] = False
        self.custom_info[pair]['trailing_buy']['trailing_buy_order_uplimit'] = 0
        self.custom_info[pair]['trailing_buy']['start_trailing_price'] = 0
        self.custom_info[pair]['trailing_buy']['buy_tag'] = None
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据特定的技术指标和条件为数据框添加买入信号，并实现追踪买入逻辑。

        参数:
        dataframe (DataFrame): 包含价格数据和技术指标的数据框。
        metadata (dict): 包含额外信息的字典。

        返回:
        DataFrame: 添加了买入信号的数据框。
        """

        def get_local_min(x):
            """
            获取滚动窗口内的局部最小值。

            参数:
            x (Series): 数据序列。

            返回:
            float: 局部最小值。
            """
            win = dataframe.loc[:, 'barssince_last_buy'].iloc[x.shape[0] - 1].astype('int')
            win = max(win, 0)
            return pd.Series(x).rolling(window=win).min().iloc[-1]

        # 调用父类的 populate_buy_trend 方法
        dataframe = super(TrailingBuyStrat, self).populate_buy_trend(dataframe, metadata)
        # 重命名 "buy" 列为 "pre_buy"
        dataframe = dataframe.rename(columns={"buy": "pre_buy"})

        # 如果启用了追踪买入且运行模式为 'live' 或 'dry_run'
        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-1].squeeze()
            if not self.process_only_new_candles:
                current_price = self.get_current_price(metadata["pair"])
            else:
                current_price = last_candle['close']
            dataframe['buy'] = 0
            # 如果尚未开始追踪买入且满足预买入条件
            if not self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] and last_candle[
                'pre_buy'] == 1:
                self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] = True
                self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price'] = last_candle['close']
                self.custom_info[metadata["pair"]]['trailing_buy']['buy_tag'] = last_candle['buy_tag']
                self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle['close']
                logger.info(f'start trailing buy for {metadata["pair"]} at {last_candle["close"]}')
            elif self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started']:
                # 更新追踪买入的上限价格
                if current_price < self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit']:
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = min(
                        current_price * (1 + self.trailing_buy_offset),
                        self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'])
                    logger.info(
                        f'update trailing buy for {metadata["pair"]} at {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                elif current_price < self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price']:
                    dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
                    ratio = "%.2f" % ((current_price / self.custom_info[metadata['pair']]['trailing_buy'][
                        'start_trailing_price']) * 100)
                    dataframe.iloc[-1, dataframe.columns.get_loc(
                        'buy_tag')] = f"{self.custom_info[metadata['pair']]['trailing_buy']['buy_tag']} ({ratio} %)"
                    # 停止追踪买入
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] = False
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = 0
                    self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price'] = None
                    self.custom_info[metadata["pair"]]['trailing_buy']['buy_tag'] = None
                else:
                    logger.info(
                        f'price too high for {metadata["pair"]} at {current_price} vs {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
        elif self.trailing_buy_order_enabled:
            # 回测模式
            dataframe.loc[
                (dataframe['pre_buy'] == 1) &
                (dataframe['pre_buy'].shift() == 0), 'pre_buy_switch'] = 1
            dataframe['pre_buy_switch'] = dataframe['pre_buy_switch'].fillna(0)
            dataframe['barssince_last_buy'] = dataframe['pre_buy_switch'].groupby(
                dataframe['pre_buy_switch'].cumsum()).cumcount()

            # 创建每行的整数位置
            idx_positions = np.arange(len(dataframe))
            # 按 shift 列的数量“移动”这些整数位置
            shifted_idx_positions = idx_positions - dataframe["barssince_last_buy"]
            # 从我们的 DatetimeIndex 中获取标签索引
            shifted_loc_index = dataframe.index[shifted_idx_positions]
            # 检索“移动”后的值并将其分配为新列
            dataframe["close_5m_last_buy"] = dataframe.loc[shifted_loc_index, "close_5m"].values

            dataframe.loc[:, 'close_lower'] = dataframe.loc[:, 'close'].expanding().apply(get_local_min)
            dataframe['close_lower'] = np.where(dataframe['close_lower'].isna() == True, dataframe['close'],
                                                dataframe['close_lower'])
            dataframe['close_lower_offset'] = dataframe['close_lower'] * (1 + self.trailing_buy_offset)
            dataframe['trailing_buy_order_uplimit'] = np.where(dataframe['barssince_last_buy'] < 20, pd.DataFrame(
                [dataframe['close_5m_last_buy'], dataframe['close_lower_offset']]).min(), np.nan)

            dataframe.loc[
                (dataframe['barssince_last_buy'] < 20) &  # 信号发出后的20根蜡烛内必须买入
                (dataframe['close'] > dataframe['trailing_buy_order_uplimit']), 'trailing_buy'] = 1

            dataframe['trailing_buy_count'] = dataframe['trailing_buy'].rolling(20).sum()

            dataframe.loc[
                (dataframe['trailing_buy'] == 1) &
                (dataframe['trailing_buy_count'] == 1), 'buy'] = 1
        else:
            # 非追踪模式下，直接根据预买入条件设置买入信号
            dataframe.loc[
                (dataframe['pre_buy'] == 1), 'buy'] = 1
        return dataframe

    def get_current_price(self, pair: str) -> float:
        """
        获取当前交易对的最新价格。

        参数:
        pair (str): 交易对。

        返回:
        float: 最新价格。
        """
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']
        return current_price
