import xgboost as xgb
import pandas as pd
import numpy as np
from common_functions.utils import model_driver
# df1 = pd.read_csv('/Users/danny/nba_bets/data/gamedf.csv', index_col=0)
# subset = df1.loc[df1['TEAM_ABBREVIATION'] == 'GSW',:]
inputlist = [('MIL','PHI'),('ATL','UTA'),('POR','LAC')]\
    # ,
    #          ('NOP','DAL'),('MIA','DEN'),('PHX','SAC'),
    #          ('ATL','GSW'),('CHA','LAL')]
md = model_driver(env = 'mac')
team_list = md.get_team_list()
game_df = md.get_df(inputlist)

