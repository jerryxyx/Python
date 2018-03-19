from downloadStocksAndVixData import sample_close, vix_close, start_date, end_date, all_weekdays, num_sample
import tensorflow as tf

vix_y = (vix_close.pct_change() > pct_threshold).shift(-1) * 1
