from pybaseball import statcast
import pandas as pd
pd.options.display.max_columns = 100
pd.set_option('future.no_silent_downcasting', True)

dates = ['2015-01-01','2016-01-01','2017-01-01','2018-01-01',
         '2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01']

for start, end in zip(dates[:-1],dates[1:]):

    year = start[:4]
    curr_year_data = statcast(start_dt=start, end_dt=end)

    curr_year_data.to_parquet(f'pitches_by_year/pitches_{year}.parquet')