from datasieve.utils import remove_outliers
from datasieve.transforms.base_transform import BaseTransform
import torch
import logging
from typing import Any, Dict

from xgboost import XGBRegressor
import time
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import pandas as pd
import scipy as spy
import numpy.typing as npt
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
import random
import optuna
# import warnings
from freqtrade.freqai.tensorboard import TBCallback
import sklearn
from optuna.samplers import TPESampler
from scipy.signal import argrelextrema
from freqtrade.freqai.utils import get_tb_logger, plot_feature_importance
from freqtrade.configuration import TimeRange
from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_seconds
from datasieve import transforms as ds
from datasieve.transforms.sklearn_wrapper import SKLearnWrapper
from datasieve.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
xgb.set_config(verbosity=0)

logger = logging.getLogger(__name__)


# Optuna
N_TRIALS = 25

"""
The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
We use sponsor money to help stimulate new features and to pay for running these public
experiments, with a an objective of helping the community make smarter choices in their
ML journey.

This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
returns. The goal is to demonstrate gratitude to people who support the project and to
help them find a good starting point for their own creativity.

If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

https://github.com/sponsors/robcaulk
"""


class XGBoostRegressorMewtwoV1(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        用户在此设置训练和测试数据，以适应他们所需的模型。

        :param data_dictionary: 包含所有训练、测试、标签和权重数据的字典。
        :param dk: FreqaiDataKitchen - 当前币种或模型的数据管理工具。
        :returns: 训练好的模型，以及用于验证的评估集。
        """

        # 从 data_dictionary 中提取训练特征和标签
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        # 检查 `data_split_parameters` 中的 `test_size` 参数
        # 如果测试集大小为 0，则没有评估集；否则，构建评估集
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            # 构建评估集，包括测试集和训练集
            eval_set = [
                (data_dictionary["test_features"], data_dictionary["test_labels"]),
                (X, y)
            ]
            # 评估集的权重
            eval_weights = [
                data_dictionary['test_weights'],
                data_dictionary['train_weights']
            ]

        # 获取训练数据的样本权重
        sample_weight = data_dictionary["train_weights"]

        # 记录训练开始时间
        start = time.time()

        # 获取初始模型或已保存的模型
        xgb_model = self.get_init_model(dk.pair)

        # 初始化 XGBoost 回归模型，传入训练参数
        model = XGBRegressor(**self.model_training_parameters)

        # 设置 XGBoost 模型的 TensorBoard 回调，用于记录训练过程数据
        model.set_params(callbacks=[TBCallback(dk.data_path)])

        # 使用评估集和样本权重训练模型
        model.fit(
            X=X, y=y, sample_weight=sample_weight,
            eval_set=eval_set, sample_weight_eval_set=eval_weights,
            xgb_model=xgb_model
        )

        # 清除回调，以便模型可以被序列化保存
        model.set_params(callbacks=[])

        # 计算训练所花费的时间
        time_spent = (time.time() - start)

        # 更新指标跟踪器以记录模型的训练时间
        self.dd.update_metric_tracker('fit_time', time_spent, dk.pair)

        # 返回训练后的模型以及评估集
        return model, eval_set

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        """
        为模型执行实时预测拟合参数和阈值。

        :param dk: FreqaiDataKitchen - 数据管理工具，提供模型训练和预测所需的数据。
        :param pair: str - 交易对（例如 "BTC/USD"）。
        """

        # 设定 warmed_up 标志，表示是否满足实时预测所需的烛台数据量
        warmed_up = True

        # 获取需要实时预测的烛台数量（默认为 100）
        num_candles = self.freqai_info.get('fit_live_predictions_candles', 100)

        # 检查当前是否处于实时模式
        if self.live:
            # 检查类中是否存在 `exchange_candles` 属性，如果不存在则初始化
            if not hasattr(self, 'exchange_candles'):
                # 设置已返回的烛台数量
                self.exchange_candles = len(self.dd.model_return_values[pair].index)

            # 计算实际烛台数量与所需烛台数量之间的差异
            candle_diff = len(self.dd.historic_predictions[pair].index) - (num_candles + self.exchange_candles)

            # 如果差异小于零，表示还没有足够的烛台数据进行实时预测
            if candle_diff < 0:
                logger.warning(f'Fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go')
                warmed_up = False

        # 获取最新的 `num_candles` 个历史预测数据并重置索引
        pred_df_full = self.dd.historic_predictions[pair].tail(num_candles).reset_index(drop=True)

        # 创建一个新的 DataFrame 来存储经过筛选的数值型数据
        pred_df_sorted = pd.DataFrame()
        for label in pred_df_full.keys():
            # 跳过非数值型的列
            if pred_df_full[label].dtype == object:
                continue
            pred_df_sorted[label] = pred_df_full[label]

        # 对每一列进行降序排序
        for col in pred_df_sorted:
            pred_df_sorted[col] = pred_df_sorted[col].sort_values(ascending=False, ignore_index=True)

        # 计算用于极值阈值的频率
        frequency = num_candles / (dk.data["kernel"] * 2)

        # 计算预测数据的最大值和最小值的平均值
        max_pred = pred_df_sorted.iloc[:int(frequency)].mean()
        min_pred = pred_df_sorted.iloc[-int(frequency):].mean()

        # 如果还没有准备好，则使用默认阈值；否则，使用计算出的最大值和最小值设置阈值
        if not warmed_up:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = 2
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = -2
        else:
            dk.data['extra_returns_per_train']['&s-maxima_sort_threshold'] = max_pred['&s-extrema']
            dk.data['extra_returns_per_train']['&s-minima_sort_threshold'] = min_pred['&s-extrema']

        # 初始化用于存储标签的均值和标准差的字典
        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}

        # 为每个标签计算均值和标准差（目前设为零）
        for ft in dk.label_list:
            dk.data['labels_std'][ft] = 0  # 标准差设为 0
            dk.data['labels_mean'][ft] = 0  # 均值设为 0

        # 拟合 DI 指标的阈值
        if not warmed_up:
            f = [0, 0, 0]
            cutoff = 2
        else:
            # 使用 Weibull 分布拟合 DI 指标值
            f = spy.stats.weibull_min.fit(pred_df_full['DI_values'])
            # 使用概率密度函数获取 99.9% 分位点作为阈值
            cutoff = spy.stats.weibull_min.ppf(0.999, *f)

        # 计算 DI 值的均值和标准差，并存储分布参数和阈值
        dk.data["DI_value_mean"] = pred_df_full['DI_values'].mean()
        dk.data["DI_value_std"] = pred_df_full['DI_values'].std()
        dk.data['extra_returns_per_train']['DI_value_param1'] = f[0]
        dk.data['extra_returns_per_train']['DI_value_param2'] = f[1]
        dk.data['extra_returns_per_train']['DI_value_param3'] = f[2]
        dk.data['extra_returns_per_train']['DI_cutoff'] = cutoff

    def train(
            self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, window: int, test_size, **kwargs
    ) -> Any:
        """
        过滤训练数据并训练模型。训练过程大量使用数据厨房来存储、保存、加载和分析数据。

        :param unfiltered_df: 当前训练周期的完整数据集。
        :param pair: str - 交易对（例如 "BTC/USD"）。
        :param dk: FreqaiDataKitchen - 数据管理工具，提供数据过滤和分析功能。
        :param window: int - 训练数据的窗口大小。
        :param test_size: float - 测试集的比例。
        :returns: 训练好的模型，可用于推理（即 `self.predict`）。
        """

        # 记录开始训练的日志
        logger.info(f"-------------------- Starting training {pair} --------------------")

        # 记录训练开始的时间戳
        start_time = time.time()

        # 使用数据厨房 `dk` 过滤特征并处理 NaN 值
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True
        )

        # 提取训练数据的起始和结束日期并记录日志
        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to {end_date} --------------------")

        # 根据给定的 `test_size` 比例划分训练和测试数据集
        dd = self.make_train_test_datasets(features_filtered, labels_filtered, dk, test_size)

        # 将训练数据截取到设定的窗口大小
        dd["train_features"] = dd["train_features"][-window:]
        dd["train_labels"] = dd["train_labels"][-window:]
        dd["train_weights"] = dd["train_weights"][-window:]

        # 如果未启用实时预测或者当前不处于实时模式，则拟合标签
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()

        # 可选的额外数据清理或分析
        dk.feature_pipeline = self.define_data_pipeline()
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        # 使用特征和标签管道清理和转换训练数据
        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(
            dd["train_features"], dd["train_labels"], dd["train_weights"]
        )
        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])

        # 如果存在测试集，使用管道转换测试数据
        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (dd["test_features"],
             dd["test_labels"],
             dd["test_weights"]) = dk.feature_pipeline.transform(
                dd["test_features"], dd["test_labels"], dd["test_weights"]
            )
            dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        # 记录用于训练的特征数和数据点数
        logger.info(f"Training model on {len(dk.data_dictionary['train_features'].columns)} features")
        logger.info(f"Training model on {len(dd['train_features'])} data points")

        # 通过调用 `fit` 函数训练模型，并获取评估集
        model, eval_set = self.fit(dd, dk)

        # 记录训练结束的时间戳并计算耗时
        end_time = time.time()
        logger.info(
            f"-------------------- Done training {pair} ({end_time - start_time:.2f} secs) --------------------")

        # 返回训练好的模型和评估集
        return model, eval_set

    def balance_training_weights(self, labels: DataFrame, weights: npt.ArrayLike,
                                 dk: FreqaiDataKitchen) -> npt.ArrayLike:
        """
        修改训练权重以强调不平衡的目标标签。
        即，当某个“类别”比其他类别数量更多时进行平衡（不仅限于分类目标）。

        :param labels: DataFrame - 包含目标标签的 DataFrame。
        :param weights: npt.ArrayLike - 原始训练权重数组。
        :param dk: FreqaiDataKitchen - 数据管理工具，提供标签列表和其他数据。
        :returns: 平衡后的权重数组。
        """

        # 获取第一个标签，用于调整平衡权重
        label = dk.label_list[0]
        logger.info(f"using {label} to balance the weights")

        # 计算绝对值以衡量目标标签的不平衡程度
        balance_weights = labels[label].abs().values.ravel()

        # 原始权重与平衡权重相加，生成新的权重数组
        weights_balanced = weights + balance_weights

        # 对新的权重数组进行归一化（缩放到 0-1 范围内）
        scaled_weights = (weights_balanced - weights_balanced.min()) / \
                         (weights_balanced.max() - weights_balanced.min())

        # 返回平衡后的权重数组
        return scaled_weights

    def make_train_test_datasets(
            self, filtered_dataframe: DataFrame, labels: DataFrame, dk: FreqaiDataKitchen, test_size: float
    ) -> Dict[Any, Any]:
        """
        使用完整训练数据集，将数据按配置文件中的用户参数划分为训练和测试数据。

        :param filtered_dataframe: 已清理、准备好分割的数据集。
        :param labels: 已清理、准备好分割的目标标签。
        :param dk: FreqaiDataKitchen - 数据管理工具，用于获取配置文件和权重设置。
        :param test_size: 测试集比例。
        :returns: 包含划分后训练和测试数据的字典。
        """

        # 获取特征参数字典
        feat_dict = dk.freqai_config["feature_parameters"]

        # 如果配置文件没有 `shuffle` 参数，默认设置为 False（不打乱顺序）
        if 'shuffle' not in dk.freqai_config['data_split_parameters']:
            dk.freqai_config["data_split_parameters"].update({'shuffle': False})

        # 初始化权重数组
        weights: npt.ArrayLike
        if feat_dict.get("weight_factor", 0) > 0:
            # 设置权重，使得最近的数据更重要
            weights = dk.set_weights_higher_recent(len(filtered_dataframe))
        else:
            # 否则，将所有样本的权重设置为 1
            weights = np.ones(len(filtered_dataframe))

        # 如果启用了权重平衡，则调用 `balance_training_weights` 函数平衡权重
        if feat_dict.get("balance_weights", False):
            weights = self.balance_training_weights(labels, weights, dk)

        # 使用 `train_test_split` 函数将数据集分割为训练和测试集
        (
            train_features,
            test_features,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        ) = train_test_split(
            filtered_dataframe[: filtered_dataframe.shape[0]],
            labels,
            weights,
            test_size=test_size,
            shuffle=False,
            random_state=1
        )

        # 检查是否需要在分割后打乱顺序
        if feat_dict["shuffle_after_split"]:
            rint1 = random.randint(0, 100)
            rint2 = random.randint(0, 100)
            # 对训练数据打乱顺序
            train_features = train_features.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_labels = train_labels.sample(frac=1, random_state=rint1).reset_index(drop=True)
            train_weights = pd.DataFrame(train_weights).sample(frac=1, random_state=rint1).reset_index(
                drop=True).to_numpy()[:, 0]
            # 对测试数据打乱顺序
            test_features = test_features.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_labels = test_labels.sample(frac=1, random_state=rint2).reset_index(drop=True)
            test_weights = pd.DataFrame(test_weights).sample(frac=1, random_state=rint2).reset_index(
                drop=True).to_numpy()[:, 0]

        # 检查是否需要反转训练和测试数据的顺序
        if dk.freqai_config['feature_parameters'].get('reverse_train_test_order', False):
            return dk.build_data_dictionary(
                test_features, train_features, test_labels, train_labels, test_weights, train_weights
            )
        else:
            return dk.build_data_dictionary(
                train_features, test_features, train_labels, test_labels, train_weights, test_weights
            )

    def extract_data_and_train_model(
            self,
            new_trained_timerange: TimeRange,
            pair: str,
            strategy: IStrategy,
            dk: FreqaiDataKitchen,
            data_load_timerange: TimeRange,
    ) -> None:
        """
        获取数据并训练模型。

        :param new_trained_timerange: TimeRange - 用于训练模型的时间范围。
        :param pair: str - 交易对，例如 "BTC/USD"。
        :param strategy: IStrategy - 用户定义的策略对象。
        :param dk: FreqaiDataKitchen - 非持久化数据容器，适用于当前交易对/循环。
        :param data_load_timerange: TimeRange - 用于加载指标数据的时间范围，应比 `new_trained_timerange` 更长，以避免 NaN 数据。
        """

        # 获取基础数据和相关数据框架
        corr_dataframes, base_dataframes = self.dd.get_base_and_corr_dataframes(
            data_load_timerange, pair, dk
        )

        # 使用策略对象填充指标，获取未过滤的数据框架
        unfiltered_dataframe = dk.use_strategy_to_populate_indicators(
            strategy, corr_dataframes, base_dataframes, pair
        )

        # 获取 `new_trained_timerange` 的结束时间戳
        trained_timestamp = new_trained_timerange.stopts

        # 初始化 TensorBoard 日志记录器
        self.tb_logger = get_tb_logger(
            self.dd.model_type, dk.data_path, self.activate_tensorboard
        )

        # 在数据容器中添加 Optuna 错误变量以供目标函数跟踪
        dk.optuna_error = None

        # 创建 Optuna study，方向为最小化
        hp = {}
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, unfiltered_dataframe, pair, dk, new_trained_timerange),
            n_trials=self.freqai_info["optuna_config"].get("n_trials", N_TRIALS),
            n_jobs=1
        )

        # 获取最佳参数并显示
        hp = study.best_params
        for key, value in hp.items():
            logger.warning(f"{key:>20s} : {value}")
        logger.info(f"{'best objective value':>20s} : {study.best_value}")

        # 使用 Optuna study 中最佳模型的相关信息
        model = dk.best_model
        dk.feature_pipeline = dk.best_data_pipeline
        dk.label_pipeline = dk.best_label_pipeline
        dk.training_features_list = dk.best_training_features_list

        # 关闭 TensorBoard 日志记录器
        self.tb_logger.close()

        # 将最佳超参数 `kernel` 和训练时间戳保存到 pair_dict
        self.dd.pair_dict[pair]["kernel"] = hp["kernel"]
        self.dd.pair_dict[pair]["trained_timestamp"] = trained_timestamp

        # 为新模型设置名称并保存模型数据
        dk.set_new_model_names(pair, trained_timestamp)
        self.dd.save_data(model, pair, dk)

        # 如果需要，绘制特征重要性图
        if self.plot_features:
            plot_feature_importance(model, pair, dk, self.plot_features)

        # 清理旧模型数据
        self.dd.purge_old_models()

    def set_freqai_targets(self, dataframe, kernel, **kwargs):
        """
        在 DataFrame 中标记用于交易的极值，并预测预期的价格范围。

        :param dataframe: DataFrame - 包含市场数据的 DataFrame。
        :param kernel: int - 用于确定相对极值的窗口大小。
        :param kwargs: dict - 额外参数以扩展或修改极值计算行为。
        :returns: 修改后的 DataFrame，包含新列标记极值及其对应的进入/退出信号。
        """

        # 在数据框中添加一个名为 "&s-extrema" 的新列，所有值初始为 0
        dataframe["&s-extrema"] = 0

        # 使用 scipy.signal 的 `argrelextrema` 函数找到 'low' 列中的相对极小值
        # `np.less` 表示寻找相对最小值，`order=kernel` 设定比较窗口大小
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=kernel
        )

        # 使用 `argrelextrema` 函数找到 'high' 列中的相对极大值
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

        # 添加 'minima-exit' 列：`&s-extrema` 列为 -1 时，设置为 1，表示买入信号
        dataframe["minima-exit"] = np.where(dataframe["&s-extrema"] == -1, 1, 0)

        # 添加 'maxima-exit' 列：`&s-extrema` 列为 1 时，设置为 1，表示卖出信号
        dataframe["maxima-exit"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)

        # 对 `&s-extrema` 列进行平滑处理，以减少噪声
        # 使用高斯窗口类型的滚动窗口，窗口大小为 5
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)

        # 预测价格范围：
        # - '&-s_max'：未来 `kernel` 个周期内最高收盘价相对于当前收盘价的变化百分比
        # - '&-s_min'：未来 `kernel` 个周期内最低收盘价相对于当前收盘价的变化百分比
        dataframe['&-s_max'] = dataframe["close"].shift(-kernel).rolling(kernel).max() / dataframe["close"] - 1
        dataframe['&-s_min'] = dataframe["close"].shift(-kernel).rolling(kernel).min() / dataframe["close"] - 1

        # 返回修改后的 DataFrame
        return dataframe

    def objective(self, trial, unfiltered_dataframe, pair, dk, timerange):
        """定义目标函数以进行模型训练和超参数优化"""

        # 获取 Optuna 配置部分
        optuna_config = self.freqai_info.get('optuna_config', {})

        # 从配置中获取用于训练周期的参数范围
        train_period_candles_min = optuna_config.get('train_period_candles_min', 1152)
        train_period_candles_max = optuna_config.get('train_period_candles_max', 11520)
        train_period_candles_step = optuna_config.get('train_period_candles_step', 2304)

        # 从配置中获取目标窗口的参数范围
        train_target_kernel_min = optuna_config.get('train_target_kernel_min', 10)
        train_target_kernel_max = optuna_config.get('train_target_kernel_max', 50)
        train_target_kernel_step = optuna_config.get('train_target_kernel_step', 10)

        # 从配置中获取测试集大小的参数范围
        test_size_min = optuna_config.get('test_size_min', 0.005)
        test_size_max = optuna_config.get('test_size_max', 0.02)
        test_size_step = optuna_config.get('test_size_step', 0.005)

        # 定义搜索空间
        window = trial.suggest_int('train_period_candles', train_period_candles_min, train_period_candles_max,
                                   step=train_period_candles_step)
        kernel = trial.suggest_int('kernel', train_target_kernel_min, train_target_kernel_max,
                                   step=train_target_kernel_step)
        test_size = trial.suggest_float('test_size', test_size_min, test_size_max, step=test_size_step)

        # 为 DataFrame 设置用于交易的极值标记和预测范围
        opt_dataframe = self.set_freqai_targets(unfiltered_dataframe, kernel)

        # 通过缓冲时间范围来确保数据足够用
        buffered_timerange = self.buffer_timerange(timerange, kernel)

        # 根据缓冲的时间范围切片数据框架
        opt_dataframe = dk.slice_dataframe(buffered_timerange, opt_dataframe)

        # 添加 Santiment 数据作为训练特征（如果需要）
        if hasattr(self, "add_santiment_data") and self.add_santiment_data:
            logger.info("Adding Santiment data to training features")
            opt_dataframe = self.add_santiment_data_to_training_features(timerange, opt_dataframe, dk)

        # 使用策略找到特征，并在数据厨房中存储
        dk.find_features(opt_dataframe)
        dk.find_labels(opt_dataframe)

        # 使用设置好的参数来训练模型
        model, eval_set = self.train(opt_dataframe, pair, dk, window, test_size)
        X_test = eval_set[0][0]
        y_test = eval_set[0][1]

        # 预测测试数据并计算误差
        y_pred = model.predict(X_test)
        error = sklearn.metrics.mean_squared_error(y_test, y_pred)

        # 如果当前误差比以前记录的最佳误差小，更新最佳模型和相关信息
        if dk.optuna_error is None or error < dk.optuna_error:
            dk.optuna_error = error
            dk.best_model = model
            dk.best_data_pipeline = dk.feature_pipeline
            dk.best_label_pipeline = dk.label_pipeline
            dk.best_training_features_list = dk.training_features_list
            dk.data["kernel"] = kernel
            dk.data["window"] = window
            dk.data["test_size"] = test_size

        # 返回误差值，用于 Optuna 目标优化
        return error

    def define_data_pipeline(self, threads=1) -> Pipeline:
        """
        定义用于数据处理的特征管道，支持自定义的 GPU 加速相异指数。

        :param threads: int - 线程数量（默认为 1）。
        :returns: 构建好的数据管道对象。
        """

        # 从 freqai_info 中获取相异指数阈值
        di = self.freqai_info["feature_parameters"].get("DI_threshold", 10)

        # 定义特征管道，包括三个处理步骤
        feature_pipeline = Pipeline([
            # `VarianceThreshold`：去除方差为 0 的特征列
            ('const', ds.VarianceThreshold(threshold=0)),

            # `MinMaxScaler`：将特征缩放到范围 -1 到 1
            ('sc', SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),

            # `DissimilarityIndex`：应用自定义相异指数，阈值为 `di`
            ('di', DissimilarityIndex(di_threshold=di))
        ])

        # 返回定义好的特征管道
        return feature_pipeline

    def buffer_timerange(self, timerange: TimeRange, buffer: int) -> TimeRange:
        """
        为时间范围添加缓冲，以扩大或缩小开始和结束时间。这在指标填充后使用。

        主要用例是在预测极大值和极小值时，`argrelextrema` 函数无法准确判断
        时间范围边缘的极大值/极小值。为了提高模型准确性，最好对整个时间范围计算极值，
        然后使用此函数按内核大小从时间范围中剪切掉边缘部分（缓冲区）。

        在另一种情况下，如果目标设置为移动价格变化，则不需要此缓冲，
        因为时间范围末端移动的烛台数据将变为 NaN，并且 FreqAI 将自动将其从训练数据集中剔除。

        :param timerange: TimeRange - 要缓冲的时间范围对象。
        :param buffer: int - 用于缩小或扩展时间范围的缓冲大小。
        :returns: 缓冲后的时间范围对象。
        """

        # 减小时间范围的结束时间，按缓冲区大小乘以每个时间单位的秒数
        timerange.stopts -= buffer * timeframe_to_seconds(self.config["timeframe"])

        # 增大时间范围的开始时间，按缓冲区大小乘以每个时间单位的秒数
        timerange.startts += buffer * timeframe_to_seconds(self.config["timeframe"])

        # 返回缓冲后的时间范围对象
        return timerange


# Keeping it here so that the file is portable and self contained, but could put this in a separate
# file and import it

class DissimilarityIndex(BaseTransform):
    def __init__(self, di_threshold: float = 1, **kwargs):
        """
        初始化 DissimilarityIndex 类，定义相异指数的阈值和其他内部变量。

        :param di_threshold: float - 相异指数阈值，用于确定哪些点被认为是异常值。
        :param kwargs: dict - 其他可选参数。
        """
        # 存储平均距离
        self.avg_mean_dist: float = 0
        # 存储训练数据的张量
        self.trained_data: torch.Tensor = torch.tensor([])
        # 设置相异指数阈值
        self.di_threshold = di_threshold
        # 存储相异指数值的数组
        self.di_values: npt.ArrayLike = np.array([])

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        计算距离，保存平均距离，并将训练数据保存为张量以供将来使用。

        :param X: 特征矩阵。
        :param y: 标签（可选）。
        :param sample_weight: 样本权重（可选）。
        :param feature_list: 特征列表（可选）。
        :param kwargs: 其他可选参数。
        :returns: 原始特征矩阵、标签、样本权重、特征列表。
        """
        # 确定计算设备（CUDA 或 CPU）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 将输入特征矩阵转换为张量，并放到设备上
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # 计算成对距离矩阵
        pairwise = torch.cdist(X_tensor, X_tensor, p=2.0)

        # 移除对角线上的距离（即自身距离），将其设置为 NaN
        pairwise.fill_diagonal_(float('nan'))
        # 将矩阵转换为一维
        pairwise = pairwise.view(-1, 1)

        # 计算平均距离，并存储
        self.avg_mean_dist = torch.nanmean(pairwise).item()
        # 保存训练数据
        self.trained_data = X_tensor

        return X, y, sample_weight, feature_list

    def transform(
        self, X, y=None, sample_weight=None, feature_list=None, outlier_check=False, **kwargs
    ):
        """
        计算每个预测点与训练数据点的距离，以估计相异指数（DI）并避免对远离训练数据集的点进行预测。

        :param X: 特征矩阵。
        :param y: 标签（可选）。
        :param sample_weight: 样本权重（可选）。
        :param feature_list: 特征列表（可选）。
        :param outlier_check: bool - 是否检查并返回异常点（默认 False）。
        :param kwargs: 其他可选参数。
        :returns: 处理后的特征矩阵、标签、样本权重、特征列表。
        """
        # 确保设备与训练数据一致
        device = self.trained_data.device
        # 将输入特征矩阵转换为张量，并放到设备上
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # 计算输入数据与训练数据之间的成对距离
        distance = torch.cdist(self.trained_data, X_tensor, p=2.0)

        # 计算相异指数值，最小距离除以平均距离
        di_values = distance.min(dim=0)[0] / self.avg_mean_dist

        # 根据相异指数阈值，标记是否超出范围（异常值）
        y_pred = torch.where(di_values < self.di_threshold, torch.tensor(1), torch.tensor(0)).cpu().numpy()
        # 存储相异指数值
        self.di_values = di_values.cpu().numpy()

        if not outlier_check:
            # 移除异常值样本
            X, y, sample_weight = remove_outliers(X, y, sample_weight, y_pred)
        else:
            # 修改标签以标记异常值
            y += y_pred
            y -= 1

        # 记录被移除的异常样本数量
        num_tossed = len(y_pred) - len(X)
        if num_tossed > 0:
            logger.info(
                f"DI tossed {num_tossed} predictions for "
                "being too far from training data."
            )

        return X, y, sample_weight, feature_list

