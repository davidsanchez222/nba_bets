import pandas as pd
import numpy as np
# from common_functions.utils import DataObject
env = 'linux'
from multiprocessing import Pool
import pandas as pd
from itertools import chain
import pandas as pd
import numpy as np
import xgboost as xgb
from multiprocessing import get_context

# helper functions

def getTeamDF(abbreviation):
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.static import teams
    all_nba_teams = pd.DataFrame(teams.get_teams())
    relevant_id = all_nba_teams[all_nba_teams['abbreviation'] == abbreviation]['id'].tolist()
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=relevant_id)
    games_dict = gamefinder.get_normalized_dict()
    games_df = pd.DataFrame(games_dict['LeagueGameFinderResults'])
    return games_df



#game object: receives the fully downloaded csv with all games
class DataObject:
    def __init__(self, inputdf):
        self.df1 = inputdf

    def getTeamStats(self,abv,latestdate):
        #edit this function for additional feature eng
        team_subset = self.df1[self.df1['TEAM_ABBREVIATION'] == abv].copy()
        # team_subset = df1[df1['TEAM_ABBREVIATION'] == 'ATL'].copy()
        # latestdate = '2000-01-19'
        team_subset['GAME_DATE'] = pd.to_datetime(team_subset['GAME_DATE'])
        team_subset.index = team_subset['GAME_DATE']
        team_subset.sort_index(inplace=True, ascending=False)
        colnames = team_subset.columns
        stats_columns = ['PTS', 'FGM', 'FGA', 'FG_PCT',
                         'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
                         'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        date_subset = team_subset[team_subset['GAME_DATE'] < latestdate].copy()
        date_subset['numerical_wins'] = np.where(date_subset['WL'] == 'L', 0, 1)
        date_subset['location'] = np.where(date_subset['MATCHUP'].str.contains('@'),-1,1)
        date_reversed = date_subset.iloc[::-1].copy()
        date_reversed['window_sum10'] = date_reversed['numerical_wins'].rolling(10).sum()
        date_reversed['window_sum5'] = date_reversed['numerical_wins'].rolling(5).sum()
        date_reversed['window_sum3'] = date_reversed['numerical_wins'].rolling(3).sum()
        stats_columns.extend(['window_sum10', 'window_sum5', 'window_sum3','location','numerical_wins'])
        date_subset = date_reversed.copy()
        current_stats = date_subset.iloc[-11:, [date_subset.columns.get_loc(c) for c in stats_columns]].copy()
        base_points = current_stats['PTS']
        current_stats['PIE'] = (
                    current_stats['PTS'] + current_stats['FGM'] + current_stats['FTM'] - current_stats[
                'FTA'] + current_stats['DREB'] +
                    current_stats['OREB'] + current_stats['AST'] + current_stats['STL'] + current_stats[
                        'BLK'] - current_stats['PF'] - current_stats['TOV'])
        current_stats['CORE_PTS'] = base_points
        current_stats.iloc[:,0:18] = current_stats.iloc[:,0:18].ewm(halflife=7).mean()
        return current_stats

    def __getSpread__(self,gameid):
        try:
            target_game = self.df1[self.df1['GAME_ID'] == gameid]  # contains target
            # target_game = df1[df1['GAME_ID'] == 29900545] #contains target
            if target_game.shape[0] != 2:
                return None
            relevant_teams = target_game['TEAM_ABBREVIATION'].tolist()
            match_location_away = target_game.loc[target_game['MATCHUP'].str.contains('@')]
            match_location_home = target_game.loc[~target_game['MATCHUP'].str.contains('@')]
            target_game_date = match_location_home['GAME_DATE']
            # match_outcome_home = np.where(match_location_away['WL'] == 'W',0,1) #0 if away team wins
            spread = match_location_home.iloc[0, match_location_home.columns.get_loc('PTS')] - \
                     match_location_away.iloc[0, match_location_away.columns.get_loc('PTS')]
            game_date = match_location_away['GAME_DATE'].values[0]
            home_team = match_location_away['MATCHUP'].str.extract(r'((?<=@.)\S{3})')[0].tolist()
            away_team = [x for x in relevant_teams if x not in home_team]
            home_df = self.getTeamStats(home_team[0], game_date)
            away_df = self.getTeamStats(away_team[0], game_date)
            normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
            normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
            if home_df.shape == (11, 25) and away_df.shape == (11, 25):
                output = [target_game_date, spread, home_df, away_df]
            else:
                return None
        except:
            return None
        return output

    def __getOverUnder__(self,gameid):
        try:
            target_game = self.df1[self.df1['GAME_ID'] == gameid]  # contains target
            # target_game = df1[df1['GAME_ID'] == 29900545] #contains target
            if target_game.shape[0] != 2:
                return None
            relevant_teams = target_game['TEAM_ABBREVIATION'].tolist()
            match_location_away = target_game.loc[target_game['MATCHUP'].str.contains('@')]
            match_location_home = target_game.loc[~target_game['MATCHUP'].str.contains('@')]
            target_game_date = match_location_home['GAME_DATE']
            # match_outcome_home = np.where(match_location_away['WL'] == 'W',0,1) #0 if away team wins
            spread = match_location_home.iloc[0, match_location_home.columns.get_loc('PTS')] + \
                     match_location_away.iloc[0, match_location_away.columns.get_loc('PTS')]
            game_date = match_location_away['GAME_DATE'].values[0]
            home_team = match_location_away['MATCHUP'].str.extract(r'((?<=@.)\S{3})')[0].tolist()
            away_team = [x for x in relevant_teams if x not in home_team]
            home_df = self.getTeamStats(home_team[0], game_date)
            away_df = self.getTeamStats(away_team[0], game_date)
            # normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
            # normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
            if home_df.shape == (11, 25) and away_df.shape == (11, 25):
                output = [target_game_date, spread, home_df, away_df]
            else:
                return None
        except:
            return None
        return output

    def get_optimization(self,label_function='spread'):
        if label_function == 'spread':
            in_func = self.__getSpread__
        elif label_function == 'over_under':
            in_func = self.__getOverUnder__
        all_games_ids = self.df1['GAME_ID'].unique()
        pool = get_context("fork").Pool(22) #change to number of cores on machine
        optimization_result = pool.map(in_func, all_games_ids)
        pool.close()
        return optimization_result


    def get_team_list(self):
        subset = self.df1[self.df1['GAME_DATE'] > '2020-01-01'].copy()
        team_list = pd.DataFrame(subset.loc[:, ['TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates())
        team_list.sort_values('TEAM_NAME',inplace=True)
        return team_list

if env == 'mac':
    root_data_dir = '/Users/danny/nba_bets/data/'
elif env == 'linux':
    root_data_dir = '/home/danny/nba/data/'

#read in data
df1 = pd.read_csv(root_data_dir + 'gamedf.csv',index_col = 0)

df1['GAME_DATE'] = pd.to_datetime(df1['GAME_DATE'])
subset_2021 = df1.loc[df1['GAME_DATE'] > '2021-01-01',['TEAM_ABBREVIATION','GAME_DATE']]

for r in range(0,subset_2021.shape[0]):
    print(subset_2021.iloc[r,0])


recent_teams = subset_2021['TEAM_ABBREVIATION'].unique()
recent_teams_df = pd.DataFrame(recent_teams)
newdf = df1.iloc[0:1000,2].value_counts()
#%%
dataset_object = DataObject(df1)
optimization_result = dataset_object.get_optimization(label_function='over_under')

complete_dataset = []
for val in optimization_result:
    if val != None :
        complete_dataset.append(val)


train_labels = []
train_features = []
test_labels = []
test_features = []
for r in range(0,len(complete_dataset)):
    print(r)
    if (pd.to_datetime(complete_dataset[r][0]) < '2020-01-01').bool():
        train_labels.append(complete_dataset[r][1])
        home_row = complete_dataset[r][2].to_numpy().flatten()
        away_row = complete_dataset[r][3].to_numpy().flatten()
        both_row = np.concatenate((home_row,away_row))
        train_features.append(both_row)
    else:
        test_labels.append(complete_dataset[r][1])
        home_row = complete_dataset[r][2].to_numpy().flatten()
        away_row = complete_dataset[r][3].to_numpy().flatten()
        both_row = np.concatenate((home_row,away_row))
        test_features.append(both_row)

train_labels = np.array(train_labels)
train_features = np.array(train_features)
test_labels = np.array(test_labels)
test_features = np.array(test_features)

#%%
import xgboost as xgb
from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(train_features, label=train_labels)


# param = {'booster':'dart', 'max_depth': 3, 'eta': .1, 'objective': 'reg:squarederror',
#          'rate_drop': 0.1,
#          'skip_drop': 0.5}
# param['tree_method'] = 'gpu_hist'
# param['sampling_method'] = 'gradient_based'
# param['eval_metric'] = 'mae'

# param = {'max_depth': 3, 'eta': .1, 'objective': 'reg:squarederror'}
# param['tree_method'] = 'gpu_hist'
# param['sampling_method'] = 'gradient_based'
# param['eval_metric'] = 'mae'
#
# num_round = 400
# bst = xgb.train(param, dtrain, num_round)


param = {'max_depth': 6, 'eta': .01, 'objective': 'reg:squarederror'}
param['tree_method'] = 'gpu_hist'
param['sampling_method'] = 'gradient_based'
param['eval_metric'] = 'mae'

num_round = 2000
bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(test_features)
predictions = bst.predict(dtest)
pred_df = pd.DataFrame([predictions,test_labels]).transpose()
pred_df.columns = ['predictions','labels']
print(mean_squared_error(pred_df['predictions'], pred_df['labels'])**.5)
print(pred_df.describe())

pred_df['label_overmean'] = np.where(pred_df['labels'] > 220.52,1,0)
pred_df['pred_overmean'] = np.where(pred_df['predictions'] > 220.52,1,0)

pred_df['correct_call'] = np.where(pred_df['label_overmean'] == pred_df['pred_overmean'], 1, 0)
print(pred_df['correct_call'].mean())

import statsmodels.api as sm
lr_df = pred_df.copy()
lr_df['const'] = 1
model = sm.OLS(lr_df['labels'], lr_df[['predictions','const']])
results = model.fit()
print(results.summary())
