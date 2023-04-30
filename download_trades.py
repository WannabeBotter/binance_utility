import os
import polars as pl
import datetime
from pathlib import Path
import re
import requests
import zipfile
from io import BytesIO
from joblib import Parallel, delayed
from retrying import retry
import argparse
import copy
import logging

from joblib_utils import tqdm_joblib
from constants import target_symbols
#from exercise_util import tqdm_joblib, identify_datafiles, target_symbols

# ファイル保存ディレクトリの中を見て、まだダウンロードしていないデータファイル名を返す関数
def identify_not_yet_downloaded_dates(symbol: str = None) -> set:
    assert symbol is not None

    _symbol = symbol.upper()
    _d_today = datetime.date.today()

    # スキャン開始日をBinanceでBTCUSDTが上場した日にする
    _d_cursor = datetime.date(year = 2019, month = 9, day = 8)
    if symbol in target_symbols:
        _initial_date = target_symbols[symbol]
        _d_cursor = datetime.date(year = _initial_date[0], month = _initial_date[1], day = _initial_date[2])
    
    _set_all_dates = set()
    while _d_cursor < _d_today:
        _set_all_dates.add(copy.copy(_d_cursor))
        _d_cursor = _d_cursor + datetime.timedelta(days = 1)
    
    _list_existing_files = Path("./data/").glob(f"{_symbol}_TRADES_*.parquet")
    _set_existing_dates = set()
    for _file_name in _list_existing_files:
        _match = re.search(r'\d{4}-\d{2}-\d{2}', _file_name.name) 
        _date_string = _match.group()
        if _date_string:
            _set_existing_dates.add(datetime.datetime.strptime(_date_string, "%Y-%m-%d"))
        
    return sorted(_set_all_dates - _set_existing_dates)

# 指定されたファイル名をもとに、.zipをダウンロードしてデータフレームを作り、pkl.gzとして保存する関数
@retry(stop_max_attempt_number = 5, wait_fixed = 1000)
def download_trades_zip(target_symbol: str = None, target_date: datetime.datetime = None) -> None:
    assert target_symbol is not None
    assert target_date is not None
    
    _target_file_name = f"{target_symbol}-trades-{target_date.strftime('%Y-%m-%d')}.zip"
    
    _stem = Path(_target_file_name).stem

    if "USD_" in target_symbol:
        _url = f'https://data.binance.vision/data/futures/cm/daily/trades/{target_symbol}/{_target_file_name}'
    else:
        _url = f'https://data.binance.vision/data/futures/um/daily/trades/{target_symbol}/{_target_file_name}'

    logging.debug(f"HTTP GET : {_url}")
    _r = requests.get(_url)
    logging.debug(f"HTTP GET status : {_r.status_code}.")
    
    if _r.status_code != requests.codes.ok:
        logging.error(f"From response.get({_url}), received HTTP status {_r.status_code}.")
        time.sleep(1)
        raise Exception
    
    _csvzip = zipfile.ZipFile(BytesIO(_r.content))
    if _csvzip.testzip() != None:
        logging.warning(f'Corrupt zip file from {_url}. Retry.')
        raise Exception
    _csvraw = _csvzip.read(f'{_stem}.csv')
    
    if chr(_csvraw[0]) == 'i':
        # ヘッダーラインがあるので削除しないといけない
        _header = 1
    else:
        _header = 0
    
    try:
        _df = pl.read_csv(BytesIO(_csvraw),
                          skip_rows = _header,
                          new_columns = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
                          dtypes = {"id": pl.Int64, "price": pl.Float64, "qty": pl.Float64, "quote_qty": pl.Float64, "time": pl.Int64, "is_buyer_maker": pl.Boolean})
        
        # カラム名などをOrderbookのデータ形式に揃える
        _df = _df.rename({"time": "timestamp", "id": "first_update_id"})
        _df = _df.with_columns(pl.col("first_update_id").alias("last_update_id"),
                               pl.when(pl.col("is_buyer_maker") == True).then("a").otherwise("b").alias("side"),
                               (pl.col("first_update_id") - 1).alias("pu"),
                               pl.lit(target_symbol).alias("symbol"),
                               pl.lit("trade").alias("update_type"))
        _df = _df.select(["symbol", "timestamp", "first_update_id", "last_update_id", "side", "update_type", "price", "qty", "pu"])
    except Exception as e:
        logging.error(f"polars.read_csv({_url}) returned Exception {e}.")
        raise e
    
    # Parquetファイルを書き込む
    Path("./data/").mkdir(parents = True, exist_ok = True)
    
    # このように一時ファイルに書き込んでからリネームしないと、ダウンロード中に強制終了した際に未完成のファイルが完全なファイルであるように見える形で残ってしまう
    _df.write_parquet(f"./data/temp_{target_symbol}_TRADES_{target_date.strftime('%Y-%m-%d')}.parquet")
    _tempfile = Path(f"./data/temp_{target_symbol}_TRADES_{target_date.strftime('%Y-%m-%d')}.parquet")
    _tempfile.rename(f"./data/{target_symbol}_TRADES_{target_date.strftime('%Y-%m-%d')}.parquet")

    return
    
# joblibを使って4並列でダウンロードジョブを実行する関数
def download_trade_from_binance(symbol: str = None) -> None:
    assert symbol is not None
    
    _symbol = symbol.upper()

    Path("./data/").mkdir(parents = True, exist_ok = True)

    # ディレクトリ内の一時ファイル一覧を取得して削除
    _file_list = [_file_path.name for _file_path in Path("./data/").glob("temp_*") if _file_path.is_file()]
    for _filename in _file_list:
        _file_obj = Path("./data/" + _filename)
        _file_obj.unlink()
    
    _set_target_dates = identify_not_yet_downloaded_dates(_symbol)  
    _num_files = len(_set_target_dates)
    logging.info(f'{symbol}の約定履歴ファイルを{_num_files}個ダウンロードします')
    
    with tqdm_joblib(total = _num_files):
        r = Parallel(n_jobs = -1, timeout = 60*60*24)([delayed(download_trades_zip)(_symbol, _target_dates) for _target_dates in _set_target_dates])

    return

# 引数処理とダウンロード関数の起動部分
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,  # ログレベルの設定
        format='%(asctime)s [%(levelname)s] %(message)s',  # ログのフォーマット設定
        handlers=[logging.FileHandler('download_trades.log', encoding='utf-8'), logging.StreamHandler()],  # ログの出力先設定
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help = 'ダウンロードする対象の銘柄 例:BTCUSDT')
    args = parser.parse_args()

    symbol = args.symbol
    if symbol:
        download_trade_from_binance(symbol)
    else:
        for _symbol in target_symbols.keys():
            download_trade_from_binance(_symbol)
