import os
import pandas as pd
import numpy as np

from src.data_preprocessing import build_targets, data_quality_report


def test_total_offsides_computation():
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01','2024-01-02']),
        'HomeTeam': ['A','B'],
        'AwayTeam': ['C','D'],
        'FTHG': [1,2], 'FTAG': [0,1],
        'HY':[1,0],'AY':[0,1],'HR':[0,0],'AR':[0,0],
        'HC':[3,4],'AC':[2,1],
        'HO':[1,2],'AO':[0,1],
    })
    out = build_targets(df)
    assert (out['total_offsides'] == pd.Series([1,3])).all()


def test_report_created(tmp_path):
    from src.config import settings
    # Redirect data dir for test
    old = settings.data_dir
    settings.data_dir = str(tmp_path)
    df = pd.DataFrame({'Date': pd.to_datetime(['2024-01-01']), 'HomeTeam':['A'], 'AwayTeam':['B']})
    data_quality_report(df)
    assert (tmp_path/ 'report.csv').exists()
    settings.data_dir = old


