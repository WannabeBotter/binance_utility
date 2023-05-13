import os
import warnings

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import joblib
import re
import polars as pl
import numpy as np
import requests
import zipfile
from io import BytesIO
from joblib import Parallel, delayed
from retrying import retry
import argparse
import logging

from tqdm import tqdm
from joblib_utils import tqdm_joblib
from constants import target_symbols

trades_dir = "./trades"
timebar_dir = "./timebar"

def calc_weighted_statistics(group_df):
    _sum_weights = group_df.select(pl.col("qty").sum())[0, 0]

    if _sum_weights == 0:
        return np.nan
    
    _weighted_mean = group_df.select((pl.col("qty") * pl.col("price")).sum() / _sum_weights)[0, 0]
    _weighted_var = group_df.select(((pl.col("qty") * (pl.col("price") - _weighted_mean) ** 2)).sum() / _sum_weights)[0, 0]
    _weighted_std = np.sqrt(_weighted_var)
    if _weighted_std != 0:
        _weighted_skewness = group_df.select(((pl.col("qty") * (pl.col("price") - _weighted_mean) ** 3)).sum() / _sum_weights)[0, 0] / _weighted_std ** 3
        _weighted_kutosis = group_df.select(((pl.col("qty") * (pl.col("price") - _weighted_mean) ** 4)).sum() / _sum_weights)[0, 0] / _weighted_std ** 4 - 3
    else:
        _weighted_skewness = 0.0
        _weighted_kutosis = 0.0
        
    return pl.DataFrame({"weighted_mean": [_weighted_mean], "weighted_var": [_weighted_var], "weighted_std": [_weighted_std], "weighted_skewness": [_weighted_skewness], "weighted_kutosis": [_weighted_kutosis]})

def calc_weighted_price_std(group_df):
    _sum_weights = group_df.select(pl.col("qty").sum())[0, 0]

    if _sum_weights == 0:
        return np.nan
    return group_df.select((pl.col("price") * pl.col("qty")).sum().alias("weighted_mean")) / _sum_weights
        
def calc_timebar_from_trades(idx, symbol, target_date, interval_sec):
    # timestampはミリ秒で入っている前提
    _year = target_date.year
    _month = target_date.month
    _day = target_date.day

    _df_trades = pl.read_parquet(f"./trades/{symbol}_TRADES_{_year:04d}-{_month:02d}-{_day:02d}.parquet")
    _df_trades = _df_trades.drop(["symbol", "first_update_id", "last_update_id", "update_type", "pu"]).sort("timestamp")
    _df_trades = _df_trades.with_columns([(pl.col("timestamp") // (interval * 1000)).cast(pl.Int64).alias("timestamp_bin"),
                                          pl.when(pl.col("side") == "a").then(-pl.col("qty")).otherwise(pl.col("qty")).alias("signed_qty")])

    # リサンプルされたOHLCを作る
    _df_groupby = _df_trades.groupby("timestamp_bin")
    _df_timebar = _df_groupby.agg(
        [pl.col("price").first().alias("open"),
         pl.col("price").max().alias("high"),
         pl.col("price").min().alias("low"),
         pl.col("price").last().alias("close"),
         pl.col("qty").sum().alias("volume"),
         pl.col("timestamp").count().alias("trade_count"),
         pl.col("signed_qty").filter(pl.col("signed_qty") >= 0).abs().sum().alias("market_bid_volume"),
         pl.col("signed_qty").filter(pl.col("signed_qty") < 0).abs().sum().alias("market_ask_volume"),
         (pl.col("signed_qty") >= 0).sum().cast(pl.Int64).alias("market_bid_count"),
         (pl.col("signed_qty") < 0).sum().cast(pl.Int64).alias("market_ask_count")])
    _df_timebar = _df_timebar.hstack(_df_groupby.apply(calc_weighted_statistics))
    _df_timebar = _df_timebar.with_columns([(pl.col("timestamp_bin") * pl.lit(1000 * interval_sec)).alias("timestamp"),
                                            (pl.col("close").log1p() - pl.col("open").log1p()).alias("log_diff_price"),
                                            (pl.col("close") - pl.col("open")).alias("delta_price")]).drop("timestamp_bin")
    # リサンプルされた際にトレードがなく空きができる場合への対策
    _timestamp_from = int(datetime(year = int(_year), month = int(_month), day = int(_day), hour = 0, minute = 0, second = 0).timestamp() * 1000)
    _timestamp_to = int((datetime(year = int(_year), month = int(_month), day = int(_day)) + timedelta(days=1)).timestamp() * 1000)
    _list_timestamps = list(range(_timestamp_from, _timestamp_to, interval_sec * 1000))
    _df_timebar = pl.DataFrame({"timestamp": _list_timestamps}).join(_df_timebar, on = "timestamp", how = "left")

    # リサンプル対象がなかった時間について、直前の値を使ってnullを埋める
    _df_timebar = _df_timebar.with_columns([pl.col("close").fill_null(strategy="forward").alias("close")])  
    _df_timebar = _df_timebar.with_columns([pl.col("open").fill_null(pl.col("close")).alias("open"),
                                            pl.col("high").fill_null(pl.col("close")).alias("high"),
                                            pl.col("low").fill_null(pl.col("close")).alias("low"),
                                            pl.col("volume").fill_null(pl.lit(0)).alias("volume"),
                                            pl.col("trade_count").fill_null(pl.lit(0)).alias("trade_count"),
                                            pl.col("market_bid_volume").fill_null(pl.lit(0)).alias("market_bid_volume"),
                                            pl.col("market_ask_volume").fill_null(pl.lit(0)).alias("market_ask_volume"),
                                            pl.col("market_bid_count").fill_null(pl.lit(0)).alias("market_bid_count"),
                                            pl.col("market_ask_count").fill_null(pl.lit(0)).alias("market_ask_count"),
                                            pl.col("weighted_mean").fill_null(pl.lit(0)).alias("weighted_mean"),
                                            pl.col("weighted_var").fill_null(pl.lit(0)).alias("weighted_var"),
                                            pl.col("weighted_std").fill_null(pl.lit(0)).alias("weighted_std"),
                                            pl.col("weighted_skewness").fill_null(pl.lit(0)).alias("weighted_skewness"),
                                            pl.col("weighted_kutosis").fill_null(pl.lit(0)).alias("weighted_kutosis"),
                                            pl.col("log_diff_price").fill_null(pl.lit(0)).alias("log_diff_price"),
                                            pl.col("delta_price").fill_null(pl.lit(0)).alias("delta_price")])
    _df_timebar = _df_timebar.select(["timestamp", "open", "high", "low", "close", "log_diff_price", "delta_price", "volume", "trade_count", "market_bid_volume", "market_ask_volume", "market_bid_count", "market_ask_count", "weighted_mean", "weighted_var", "weighted_std", "weighted_skewness", "weighted_kutosis"])

    # Parquetファイルを書き込む
    Path(timebar_dir).mkdir(parents = True, exist_ok = True)

    # 1行目のOpenがNaNの場合は、全ての時間足ファイルの生成が終わってから前日Closeを使ってOpenを埋める必要があるので、ファイル名でマークしておく
    if _df_timebar["close"].is_null()[0]:
        _parquet_filename = f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.strftime('%Y-%m-%d')}.parquet.incomplete"
    else:
        _parquet_filename = f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.strftime('%Y-%m-%d')}.parquet"
        
    # このように一時ファイルに書き込んでからリネームしないと、ダウンロード中に強制終了した際に未完成のファイルが完全なファイルであるように見える形で残ってしまう
    _df_timebar.write_parquet(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.strftime('%Y-%m-%d')}.parquet.temp", compression="zstd", compression_level=8)
    _tempfile = Path(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.strftime('%Y-%m-%d')}.parquet.temp")
    _tempfile.rename(_parquet_filename)

    return idx

# Incompleteなタイムバーファイルを完成させる関数
def finish_incomplete_timebar_files(idx, symbol, target_date, interval_sec):
    _df_incomplete_timebar = pl.read_parquet(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.strftime('%Y-%m-%d')}.parquet.incomplete")

    _previous_date = target_date - timedelta(days = 1)
    
    _previous_completed_file = Path(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{_previous_date.year:04}-{_previous_date.month:02}-{_previous_date.day:02}.parquet")
    _previous_incomplete_file = Path(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{_previous_date.year:04}-{_previous_date.month:02}-{_previous_date.day:02}.parquet.incomplete")

    _target_file = None
    if _previous_completed_file.exists() == True:
        _target_file = _previous_completed_file
    elif _previous_incomplete_file.exists() == True:
        _target_file = _previous_incomplete_file

    if _target_file is not None:
        try:
            _df_previous_date = pl.read_parquet(str(_target_file))
        except Exception as e:
            print(f'ファイル {_target_file}を読み込み中に例外{e}が発生しました')
            raise e
        
        _last_close = _df_previous_date.select(pl.col("close"))[-1, 0]
    else:
        # このファイルがこの銘柄の最初の日の記録なので、最終クローズは0とする
        _last_close = 0.0

    _df_incomplete_timebar[0, "open"] = _last_close
    _df_incomplete_timebar[0, "high"] = _last_close
    _df_incomplete_timebar[0, "low"] = _last_close
    _df_incomplete_timebar[0, "close"] = _last_close
    _df_incomplete_timebar = _df_incomplete_timebar.with_columns([pl.col("open").fill_null(strategy="forward").alias("open"),
                                                                  pl.col("high").fill_null(strategy="forward").alias("high"),
                                                                  pl.col("low").fill_null(strategy="forward").alias("low"),
                                                                  pl.col("close").fill_null(strategy="forward").alias("close")])

    # 並列処理している他のプロセスが書き込み途中のファイルを読み込まないように、一時ファイルに保存する
    _df_incomplete_timebar.write_parquet(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.year:04}-{target_date.month:02}-{target_date.day:02}.parquet.temp")
    _tempfile = Path(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.year:04}-{target_date.month:02}-{target_date.day:02}.parquet.temp")
    _tempfile.rename(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.year:04}-{target_date.month:02}-{target_date.day:02}.parquet")
    _incompletefile = Path(f"{timebar_dir}/{symbol}_TIMEBAR_{interval_sec}SEC_{target_date.strftime('%Y-%m-%d')}.parquet.incomplete")
    _incompletefile.unlink()
    

    return idx

# 全コア数-2個のコアで並列処理を行い、価格ファイルを処理して約定プロファイルを作成する関数
def generate_timebar_files(symbol: str = None, interval: int = None):
    assert symbol is not None
    assert interval is not None

    _symbol = symbol.upper()

    # 処理開始前に全てのincomplete / tempファイルを削除する
    for _incomplete_file in Path(timebar_dir).glob(f"{_symbol}_TIMEBAR_*.parquet.incomplete"):
        _incomplete_file.unlink()
    for _temp_file in Path(timebar_dir).glob(f"{_symbol}_TIMEBAR_*.parquet.temp"):
        _temp_file.unlink()
    
    # タイムバーを生成する (この時点ではまだ一日の始まりのタイムバーのOpenがNaNで、ファイル名先頭にincomplete-がついているものが存在する可能性がある)
    logging.info(f'{symbol}の{interval}秒タイムバーファイルを約定履歴から生成します')

    # トレードファイルの日付一覧を取得する
    _list_trade_filename = [str(_) for _ in Path(trades_dir).glob(f"{_symbol}_TRADES_*.parquet")]
    _list_trade_datetime = [datetime.strptime(_match.group(), "%Y-%m-%d") for _filename in _list_trade_filename if (_match := re.search(r"\d{4}-\d{2}-\d{2}", _filename))]
    _set_trade_datetime = set(_list_trade_datetime)

    # すでに完成しているタイムバーファイルの日付一覧を取得する
    _list_timebar_filename = [str(_) for _ in Path(timebar_dir).glob(f"{_symbol}_TIMEBAR_{interval}SEC_*.parquet")]
    _list_timebar_datetime = [datetime.strptime(_match.group(), "%Y-%m-%d") for _filename in _list_timebar_filename if (_match := re.search(r"\d{4}-\d{2}-\d{2}", _filename))]
    _set_timebar_datetime = set(_list_timebar_datetime)

    _set_missing_timebar_datetime = _set_trade_datetime - _set_timebar_datetime
    _list_missing_timebar_datetime = sorted(list(_set_missing_timebar_datetime))
    _num_rows = len(_list_missing_timebar_datetime)

    with tqdm_joblib(total = _num_rows):
        results = joblib.Parallel(n_jobs = -2, timeout = 60*60*24)([joblib.delayed(calc_timebar_from_trades)(_idx, _symbol, _datetime, interval) for _idx, _datetime in enumerate(_list_missing_timebar_datetime)])
    

    # Incompleteなタイムバーファイルの日付一覧を取得する
    _list_incomplete_filename = [str(_) for _ in Path(timebar_dir).glob(f"{_symbol}_TIMEBAR_{interval}SEC_*.parquet.incomplete")]
    _list_incomplete_datetime = [datetime.strptime(_match.group(), "%Y-%m-%d") for _filename in _list_incomplete_filename if (_match := re.search(r"\d{4}-\d{2}-\d{2}", _filename))]
    _num_rows = len(_list_incomplete_datetime)
    with tqdm_joblib(total = _num_rows):
        results = joblib.Parallel(n_jobs = -1, timeout = 60*60*24)([joblib.delayed(finish_incomplete_timebar_files)(_idx, _symbol, _datetime, interval) for _idx, _datetime in enumerate(_list_incomplete_datetime)])

# 引数処理とダウンロード関数の起動部分
if __name__ == '__main__':
    # カレントディレクトリの設定
    _script_path = Path(__file__).resolve()
    os.chdir(_script_path.parent)

    # ログ関係の設定
    logging.basicConfig(
        level=logging.DEBUG,  # ログレベルの設定
        format='%(asctime)s [%(levelname)s] %(message)s',  # ログのフォーマット設定
        handlers=[logging.FileHandler('download_trades.log', encoding='utf-8'), logging.StreamHandler()],  # ログの出力先設定
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help = 'ダウンロードする対象の銘柄 例:BTCUSDT')
    parser.add_argument('interval', type = int, help = '生成するタイムバーの時間間隔 [秒] 例:60')
    args = parser.parse_args()

    symbol = args.symbol
    interval = args.interval

    if 86400 % interval != 0:
        print('interval は 86400秒 (1日) の約数を指定してください')
        exit(0)
         
    if symbol:
        generate_timebar_files(symbol, int(interval))
    else:
        for _symbol in target_symbols.keys():
            generate_timebar_files(_symbol, int(interval))
