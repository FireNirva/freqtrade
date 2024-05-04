import logging
from functools import reduce
import datetime
from datetime import timedelta
import talib.abstract as ta
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from scipy.signal import argrelextrema
import numpy as np
import pandas_ta as pta
import math

logger = logging.getLogger(__name__)


class MewV11(IStrategy):
    position_adjustment_enable = False

    # Attempts to handle large drops with DCA. High stoploss is required.
    stoploss = -0.02

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
    }

    # # Example specific variables
    max_entry_position_adjustment = 1
    # # This number is explained a bit further down
    max_dca_multiplier = 2

    minimal_roi = {"0": 0.04, "5000": -1}

    process_only_new_candles = True

    can_short = True

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "&s-extrema": {
                    "color": "#f53580",
                    "type": "line"
                },
                "&s-minima_sort_threshold": {
                    "color": "#4ae747",
                    "type": "line"
                },
                "&s-maxima_sort_threshold": {
                    "color": "#5b5e4b",
                    "type": "line"
                }
            },
            "min_max": {
                "maxima-exit": {
                    "color": "#a29db9",
                    "type": "bar"
                },
                "minima-exit": {
                    "color": "#ac7fc",
                    "type": "bar"
                }
            },
            "range_est": {
                "&-s_max": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "&-s_min": {
                    "color": "#ac7fc",
                    "type": "line"
                }
            }
        }
    }

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            },

        ]

    use_exit_signal = True
    startup_candle_count: int = 80
    # # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        """
        Enrich the DataFrame with a comprehensive set of technical indicators, each adjusted by a specified period.
        This method applies a wide range of technical analysis tools to provide various metrics
        that aid in analyzing market conditions.

        :param dataframe: DataFrame - The DataFrame containing market data such as OHLC and volume.
        :param period: int - The time period over which the indicators are calculated.
        This impacts the sensitivity and the lookback span of each indicator.
        :param kwargs: dict - Additional keyword arguments for custom processing.

        Returns:
        DataFrame - The enhanced DataFrame with additional technical indicators calculated over the specified period.
        """
        # Relative Strength Index
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        # Money Flow Index
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        # Average Directional Index
        dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
        # Commodity Channel Index
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        # Efficiency Ratio
        dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
        # Rate of Change Ratio
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        # Chaikin Money Flow
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        # Top Percent Change
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        # Cyclical Turning Point
        dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
        # Choppiness Index
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        # Linear Regression Angle
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=period)
        # Average True Range
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        # Average True Range Percentage
        dataframe["%-atr-periodp"] = dataframe[f"%-atr-period"] / \
                                     dataframe['close'] * 1000
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        """
        Enrich the DataFrame with a variety of financial indicators and measurements to aid in trading decisions.
        This includes creating features based on percentage changes, volume, price distances from moving averages,
        Bollinger bands, MACD values, VWAP bands, pivot points, and more.

        :param dataframe: DataFrame - The DataFrame containing market data such as OHLC and volume.
        :param kwargs: dict - Additional keyword arguments for custom processing.

        Returns:
        DataFrame - The enhanced DataFrame with additional technical indicators and calculated metrics.
        """
        # Calculate percentage change in closing prices
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        # Store raw volume data
        dataframe["%-raw_volume"] = dataframe["volume"]
        # Calculate On-Balance Volume
        dataframe["%-obv"] = ta.OBV(dataframe)
        # Calculate Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=14, stds=2.2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        # Bollinger Band width as a percentage
        dataframe["%-bb_width"] = (dataframe["bb_upperband"] -
                                   dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        # Calculate Exponential Moving Averages
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        # Distance to EMAs
        dataframe['%-distema50'] = get_distance(
            dataframe['close'], dataframe['ema_50'])
        dataframe['%-distema12'] = get_distance(
            dataframe['close'], dataframe['ema_12'])
        dataframe['%-distema26'] = get_distance(
            dataframe['close'], dataframe['ema_26'])
        # Calculate MACD
        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd']
        dataframe['%-macdsignal'] = macd['macdsignal']
        dataframe['%-macdhist'] = macd['macdhist']
        # Distance to MACD signal and zero histogram
        dataframe['%-dist_to_macdsignal'] = get_distance(
            dataframe['%-macd'], dataframe['%-macdsignal'])
        dataframe['%-dist_to_zerohist'] = get_distance(
            0, dataframe['%-macdhist'])
        # Calculate VWAP bands
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        # VWAP band width as a percentage
        dataframe['%-vwap_width'] = ((dataframe['vwap_upperband'] -
                                      dataframe['vwap_lowerband']) / dataframe['vwap_middleband']) * 100
        # Distance to VWAP bands
        dataframe = dataframe.copy()
        dataframe['%-dist_to_vwap_upperband'] = get_distance(
            dataframe['close'], dataframe['vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(
            dataframe['close'], dataframe['vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(
            dataframe['close'], dataframe['vwap_lowerband'])
        # Candle tail and wick sizes
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        # Calculate pivot points
        pp = pivots_points(dataframe)
        dataframe['pivot'] = pp['pivot']
        dataframe['r1'] = pp['r1']
        dataframe['s1'] = pp['s1']
        dataframe['r2'] = pp['r2']
        dataframe['s2'] = pp['s2']
        dataframe['r3'] = pp['r3']
        dataframe['s3'] = pp['s3']
        # Distance to pivot points
        dataframe['rawclose'] = dataframe['close']
        dataframe['%-dist_to_r1'] = get_distance(
            dataframe['close'], dataframe['r1'])
        dataframe['%-dist_to_r2'] = get_distance(
            dataframe['close'], dataframe['r2'])
        dataframe['%-dist_to_r3'] = get_distance(
            dataframe['close'], dataframe['r3'])
        dataframe['%-dist_to_s1'] = get_distance(
            dataframe['close'], dataframe['s1'])
        dataframe['%-dist_to_s2'] = get_distance(
            dataframe['close'], dataframe['s2'])
        dataframe['%-dist_to_s3'] = get_distance(
            dataframe['close'], dataframe['s3'])
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]
        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        """
        Enhance the DataFrame with time-based features that are normalized and transformed into cyclical features using
        trigonometric functions. This approach helps in capturing the cyclical nature of days and hours which could
        influence market behavior.

        :param dataframe: DataFrame - The DataFrame containing market data, specifically requiring a 'date' column.
        :param kwargs: dict - Additional keyword arguments for custom processing.

        Returns:
        DataFrame - The DataFrame enhanced with new cyclical features based on day of the week and hour of the day.
        """
        dataframe["day_of_week"] = (dataframe["date"].dt.dayofweek)
        dataframe["hour_of_day"] = (dataframe["date"].dt.hour)
        dataframe['day_of_week_norm'] = 2 * math.pi * \
                                        dataframe['day_of_week'] / dataframe['day_of_week'].max()
        dataframe['hour_of_day_norm'] = 2 * math.pi * \
                                        dataframe['hour_of_day'] / dataframe['hour_of_day'].max()

        dataframe['%%-day_of_week_cos'] = np.cos(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_cos'] = np.cos(dataframe['hour_of_day_norm'])
        dataframe['%%-day_of_week_sin'] = np.sin(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_sin'] = np.sin(dataframe['hour_of_day_norm'])
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        """
        Define and label extremas as potential targets for entering or exiting trades within the DataFrame.
        This function calculates minima and maxima in the market data to signal optimal points for market entries and exits.

        :param dataframe: DataFrame - The DataFrame containing market data.
        :param kwargs: dict - Additional keyword arguments used to adjust the behavior of extremas calculation.

        Returns:
        DataFrame - The modified DataFrame with extremas and corresponding entry/exit signals populated.
        """
        # 在数据框中添加一个名为 "&s-extrema" 的新列，所有值初始为0
        dataframe["&s-extrema"] = 0

        # 获取用于识别极值的窗口大小（或称“内核”）
        kernel = self.freqai_info["feature_parameters"]["label_period_candles"]

        # 使用 scipy.signal 的 argrelextrema 来找出 'low' 列中的相对极小值
        # `np.less` 表示寻找相对最小值，`order=kernel` 设定比较窗口大小
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=kernel
        )

        # 同样使用 argrelextrema 来找出 'high' 列中的相对极大值
        # `np.greater` 表示寻找相对最大值，`order=kernel` 设定比较窗口大小
        max_peaks = argrelextrema(
            dataframe["high"].values, np.greater,
            order=kernel
        )

        # 遍历相对极小值的索引，将相应位置的 `&s-extrema` 列设为 -1，表示买入信号
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1

        # 遍历相对极大值的索引，将相应位置的 `&s-extrema` 列设为 1，表示卖出信号
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1

        # 新增 'minima-exit' 列，当 `&s-extrema` 列为 -1 时，设置为 1，表示潜在买入信号
        dataframe["minima-exit"] = np.where(
            dataframe["&s-extrema"] == -1, 1, 0)

        # 新增 'maxima-exit' 列，当 `&s-extrema` 列为 1 时，设置为 1，表示潜在卖出信号
        dataframe["maxima-exit"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)

        # 对 `&s-extrema` 列进行平滑处理，以减少噪声
        # 使用高斯窗口（`win_type='gaussian'`）的滚动窗口
        # `window=5` 指定平滑窗口大小，`std=0.5` 指定高斯标准偏差
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)

        # 返回修改后的 DataFrame
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate additional indicators into the DataFrame to support entry and exit strategies.

        :param dataframe: DataFrame - The input dataframe containing market data and indicators.
        :param metadata: dict - A dictionary of metadata providing additional context for strategy execution.

        Returns:
        DataFrame - The modified DataFrame with new indicators and thresholds added.
        """
        # 使用 freqai 模块的 start 方法，在当前 DataFrame 中填充所需的初始指标和数据。
        dataframe = self.freqai.start(dataframe, metadata, self)

        # 新增 "DI_catch" 列，根据 "DI_values" 列的值与 "DI_cutoff" 阈值的比较结果
        # 当 "DI_values" 大于 "DI_cutoff" 时，"DI_catch" 列值设为 0（不满足条件），否则设为 1（满足条件）
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )

        # 将 "&s-minima_sort_threshold" 列重命名为 "minima_sort_threshold" 以保持数据一致性
        dataframe["minima_sort_threshold"] = dataframe["&s-minima_sort_threshold"]

        # 将 "&s-maxima_sort_threshold" 列重命名为 "maxima_sort_threshold" 以保持数据一致性
        dataframe["maxima_sort_threshold"] = dataframe["&s-maxima_sort_threshold"]

        # 返回添加新指标和阈值的 DataFrame
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals based on defined conditions within the DataFrame. This method evaluates conditions for
        both long and short entries and marks the DataFrame accordingly.

        :param df: DataFrame - The DataFrame containing market data with indicators that may
         be used to determine entry conditions.
        :param metadata: dict - A dictionary of metadata providing additional context
        (like pair, timeframe, etc.) for strategy execution.

        Returns:
        DataFrame - The modified DataFrame that includes new columns indicating whether to enter long or short trades
        based on the evaluated conditions.
        """

        # 初始化一个空字符串的 'enter_tag' 列，用于标记交易方向（'long' 或 'short'）
        df['enter_tag'] = ''

        # 多头进入条件集合
        enter_long_conditions = [
            df["do_predict"] == 1,  # 预测信号符合要求
            df["DI_catch"] == 1,  # DI 指标符合要求
            df["&s-extrema"] < df["minima_sort_threshold"]  # 极值小于指定的多头阈值
        ]

        # 如果多头条件存在且满足，标记对应的行
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),  # 所有条件都满足
                ["enter_long", "enter_tag"]
            ] = (1, "long")  # 设置 'enter_long' 列为 1，同时 'enter_tag' 标记为 'long'

        # 空头进入条件集合
        enter_short_conditions = [
            df["do_predict"] == 1,  # 预测信号符合要求
            df["DI_catch"] == 1,  # DI 指标符合要求
            df["&s-extrema"] > df["maxima_sort_threshold"]  # 极值大于指定的空头阈值
        ]

        # 如果空头条件存在且满足，标记对应的行
        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),  # 所有条件都满足
                ["enter_short", "enter_tag"]
            ] = (1, "short")  # 设置 'enter_short' 列为 1，同时 'enter_tag' 标记为 'short'

        # 返回修改后的 DataFrame
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit trend indicators or conditions on the DataFrame. This method can be modified to include logic
        for exiting trades based on custom conditions defined within the strategy.

        :param df: DataFrame - The DataFrame containing market data with indicators
        that may be used to determine exit conditions.
        :param metadata: dict - A dictionary of metadata providing additional context (like pair, timeframe, etc.)
        for strategy execution.

        Returns:
        DataFrame - The modified DataFrame possibly including new columns that signal potential exit conditions.
        """
        return df

    def custom_exit(
            self,
            pair: str,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            **kwargs
    ):
        """
        Custom exit logic to determine if a trade should be closed based on specific market conditions
        and trade characteristics.

        :param pair: str - The trading pair (e.g., "BTC/USD").
        :param trade: Trade - The trade object containing details like entry time.
        :param current_time: datetime - The current timestamp at which the exit condition is being evaluated.
        :param current_rate: float - The current rate of the trading pair.
        :param current_profit: float - The current profit or loss of the trade.
        :param kwargs: dict - Additional keyword arguments for advanced handling.

        Returns:
        str or None - A string indicating the reason for the trade exit, or None if the trade should continue.
        """
        # 获取与交易对和时间框架对应的 DataFrame 以及元数据
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )

        # 获取 DataFrame 中的最后一根蜡烛数据
        last_candle = dataframe.iloc[-1].squeeze()

        # 确定与交易开始时间对应的蜡烛数据时间戳
        trade_date = timeframe_to_prev_date(
            self.timeframe,
            (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        )

        # 在 DataFrame 中获取交易对应的蜡烛数据
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        if trade_candle.empty:
            # 如果无法找到与交易日期对应的蜡烛数据，退出并返回 None
            return None

        # 将 `trade_candle` 压缩为一个 Series
        trade_candle = trade_candle.squeeze()

        # 获取交易的进入标签（'long' 或 'short'）
        entry_tag = trade.enter_tag

        # 计算交易持续时间（分钟）
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # 如果交易持续时间超过 1000 分钟，返回 'trade expired'，表示交易超时
        if trade_duration > 1000:
            return "trade expired"

        # 如果最后一根蜡烛的 DI_catch 指标为 0，返回 'Outlier detected'
        if last_candle["DI_catch"] == 0:
            return "Outlier detected"

        # 对空头交易：
        # 如果最后一根蜡烛的 &s-extrema 小于 minima_sort_threshold，且交易标签为 'short'，返回 'minimia_detected_short'
        if (
                last_candle["&s-extrema"] < last_candle["minima_sort_threshold"]
                and entry_tag == "short"
        ):
            return "minimia_detected_short"

        # 对多头交易：
        # 如果最后一根蜡烛的 &s-extrema 大于 maxima_sort_threshold，且交易标签为 'long'，返回 'maxima_detected_long'
        if (
                last_candle["&s-extrema"] > last_candle["maxima_sort_threshold"]
                and entry_tag == "long"
        ):
            return "maxima_detected_long"

    def confirm_trade_entry(
            self,
            pair: str,
            order_type: str,
            amount: float,
            rate: float,
            time_in_force: str,
            current_time: datetime,
            entry_tag: Optional[str],
            side: str,
            **kwargs
    ) -> bool:
        """
        Determine whether a new trade entry should be made based on the current
        market conditions and the state of open trades.

        :param pair: str - The trading pair (e.g., "BTC/USD").
        :param order_type: str - The type of order (e.g., "limit", "market").
        :param amount: float - The amount of the asset to trade.
        :param rate: float - The rate at which the trade is to be executed.
        :param time_in_force: str - The duration for which the order remains active.
        :param current_time: datetime - The current timestamp at which the trade decision is being made.
        :param entry_tag: Optional[str] - A tag associated with the entry strategy (e.g., "long", "short").
        :param side: str - The side of the trade ("long" for buying, "short" for selling).
        :param kwargs: dict - Additional keyword arguments that might be required for advanced strategies.

        Returns:
        bool - True if the trade can be executed, False otherwise.
        """
        # 获取当前所有处于开仓状态的交易
        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        # 初始化多头和空头交易计数器
        num_shorts, num_longs = 0, 0

        # 遍历所有开仓交易，按标签统计多头和空头数量
        for trade in open_trades:
            if "short" in trade.enter_tag:
                num_shorts += 1
            elif "long" in trade.enter_tag:
                num_longs += 1

        # 限制多头交易数量最多为 5，超过时不允许新开多头
        if side == "long" and num_longs >= 5:
            return False

        # 限制空头交易数量最多为 5，超过时不允许新开空头
        if side == "short" and num_shorts >= 5:
            return False

        # 从指定的交易对和时间框架中获取分析后的 DataFrame
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # 获取 DataFrame 中的最后一根蜡烛数据
        last_candle = df.iloc[-1].squeeze()

        # 检查当前的交易价格是否符合一定的价差规则
        if side == "long":
            # 如果当前买入价格比最后收盘价高 0.25% 以上，则不允许执行交易
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            # 如果当前卖出价格比最后收盘价低 0.25% 以上，则不允许执行交易
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        # 满足所有条件时允许执行交易
        return True


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Calculate the percentage change between the highest open price in the past 'length' candles
    and the current close price.
    :param dataframe: DataFrame - The input dataframe containing the OHLC data.
    :param length: int - The number of candles to look back to find the maximum open price.
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


def chaikin_mf(df, periods=20):
    """
    Calculate the Chaikin Money Flow indicator, measuring the volume-weighted average of accumulation
    and distribution over a specified period.
    :param df: DataFrame - The input dataframe containing the OHLC data and volume.
    :param periods: int - The number of periods to consider for the rolling sum and volume.
    """
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')


# VWAP bands


def VWAPB(dataframe, window_size=20, num_of_std=1):
    """
    Calculate the Volume Weighted Average Price (VWAP) and its upper and lower bands using standard deviation adjustments.
    :param dataframe: DataFrame - The input dataframe containing the OHLC data and volume.
    :param window_size: int - The number of periods used to calculate the VWAP.
    :param num_of_std: int - The number of standard deviations to determine the width of the bands.
    """
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def EWO(dataframe, sma_length=5, sma2_length=35):
    """
    Calculate the Elliott Wave Oscillator which is the difference between two exponential moving averages, scaled to the closing price.
    This indicator is used to identify waves in the price movements of a security.

    :param dataframe: DataFrame - The input dataframe containing the OHLC data.
    :param sma_length: int - The period of the shorter exponential moving average.
    :param sma2_length: int - The period of the longer exponential moving average.

    Returns:
    Series - A pandas Series object representing the Elliott Wave Oscillator values.
    """
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def get_distance(p1, p2):
    """
    Calculate the absolute distance between two points.
    :param p1: float - First point.
    :param p2: float - Second point.
    """
    return abs((p1) - (p2))
