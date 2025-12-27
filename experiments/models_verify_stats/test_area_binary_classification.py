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

def getTeamStats(abv, latestdate):
    # edit this function for additional feature eng
    team_subset = df1[df1['TEAM_ABBREVIATION'] == abv].copy()
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
    date_subset['location'] = np.where(date_subset['MATCHUP'].str.contains('@'), -1, 1)
    date_reversed = date_subset.iloc[::-1].copy()
    date_reversed['window_sum10'] = date_reversed['numerical_wins'].rolling(10).sum()
    date_reversed['window_sum5'] = date_reversed['numerical_wins'].rolling(5).sum()
    date_reversed['window_sum3'] = date_reversed['numerical_wins'].rolling(3).sum()
    stats_columns.extend(['window_sum10', 'window_sum5', 'window_sum3', 'location', 'numerical_wins','days_since'])
    date_subset = date_reversed.copy()
    date_subset_og_index = pd.Series(date_subset.index)
    date_subset_new_index = pd.Series(date_subset.index)
    date_subset_combined = pd.concat([date_subset_og_index,date_subset_new_index],axis=1)
    date_subset_combined['shifted'] = date_subset_combined.iloc[:,1].shift(1)
    date_subset_combined.columns = ['original','new','shifted']
    date_subset_combined['days_since'] = date_subset_combined['original'] - date_subset_combined['shifted']
    date_subset_combined.set_index('original',inplace=True)
    date_subset_combined.drop(['new','shifted'],axis=1,inplace=True)
    date_subset = pd.concat([date_subset,date_subset_combined],axis=1)
    date_subset['days_since'] = date_subset['days_since'].dt.days

    current_stats = date_subset.iloc[-11:, [date_subset.columns.get_loc(c) for c in stats_columns]].copy()
    current_stats['days_since'] = current_stats['days_since'].astype(int)
    base_points = current_stats['PTS']
    current_stats['PIE'] = (
            current_stats['PTS'] + current_stats['FGM'] + current_stats['FTM'] - current_stats[
        'FTA'] + current_stats['DREB'] +
            current_stats['OREB'] + current_stats['AST'] + current_stats['STL'] + current_stats[
                'BLK'] - current_stats['PF'] - current_stats['TOV'])
    current_stats['CORE_PTS'] = base_points
    current_stats.iloc[:, 0:18] = current_stats.iloc[:, 0:18].ewm(halflife=7).mean()
    current_stats['relative_performance'] = (current_stats['CORE_PTS'] - current_stats['PTS']) * (current_stats['location']+2)
    return current_stats

def getBinaryRecords(gameid):
    try:
        target_game = df1[df1['GAME_ID'] == gameid]  # contains target
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
        home_df = getTeamStats(home_team[0], game_date)
        away_df = getTeamStats(away_team[0], game_date)
        # normalized_hdf = (home_df - home_df.min()) / (home_df.max() - home_df.min())
        # normalized_adf = (away_df - away_df.min()) / (away_df.max() - away_df.min())
        if home_df.shape == (11, 27) and away_df.shape == (11, 27):
            pass
        else:
            return None
        home_df.reset_index(drop=True,inplace=True)
        away_df.reset_index(drop=True, inplace=True)
        combined_df = pd.concat([home_df,away_df],axis=1)
        ##
        val = 0
        for c in range(0,combined_df.shape[1]):
            for r in range(0,combined_df.shape[0]):
                combined_df.iloc[r,c] = val
                val = val+1
        ##
        features_row = combined_df.unstack().to_frame().reset_index(drop=True).T
        barometer = np.arange(165, 243, 1)
        target_game_date = target_game_date.reset_index(drop=True)
        output = pd.DataFrame()
        for i in barometer:
            # print(i)
            current_gauge = i
            if current_gauge < spread:
                target_label = pd.Series(0)
            else:
                target_label = pd.Series(1)
            new_features = pd.concat([pd.Series(current_gauge),features_row],axis=1)
            outrow = pd.concat([target_game_date,target_label,new_features],axis=1)
            output = pd.concat([output,outrow])
        outdf = pd.DataFrame(output)
    except:
        return None
    return outdf


if env == 'mac':
    root_data_dir = '/Users/danny/nba_bets/data/'
elif env == 'linux':
    root_data_dir = '/home/danny/nba/data/'

#read in data
df1 = pd.read_csv(root_data_dir + 'gamedf.csv',index_col = 0)
#%%
all_games_ids = df1['GAME_ID'].unique()
pool = get_context("fork").Pool(22) #change to number of cores on machine
optimization_result = pool.map(getBinaryRecords, all_games_ids)
pool.close()

complete_dataset = pd.concat(optimization_result)


#%%
train_df = complete_dataset[complete_dataset['GAME_DATE'] < '2020-01-01']
test_df = complete_dataset[complete_dataset['GAME_DATE'] >= '2020-01-01']


train_labels = train_df.iloc[:,1]
train_features = train_df.iloc[:,2:]
test_labels = test_df.iloc[:,1]
test_features = test_df.iloc[:,2:]

subset = pd.DataFrame(train_features.iloc[0,:])


train_labels = np.array(train_labels)
train_features = np.array(train_features)
test_labels = np.array(test_labels)
test_features = np.array(test_features)

#%%
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dtrain = xgb.DMatrix(train_features, label=train_labels)


param = {'max_depth': 6, 'eta': .1, 'objective': 'binary:logistic'}
param['tree_method'] = 'gpu_hist'
param['sampling_method'] = 'gradient_based'
param['eval_metric'] = 'mae'

num_round = 300
bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(test_features)
predictions = bst.predict(dtest)
pred_df = pd.DataFrame([predictions,test_labels]).transpose()
pred_df.columns = ['predictions','labels']
pred_df['binary'] = np.where(pred_df['predictions'] < .5, 0,1)

print(accuracy_score(pred_df['labels'], pred_df['binary']))
print(classification_report(pred_df['labels'], pred_df['binary']))

bst.get_score(importance_type='weight')
feature_important = bst.get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

bst.save_model(root_data_dir + 'overunder_binary.bst')

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
plt.show()

#%%
bst_overunder = xgb.Booster()
bst_overunder.load_model(root_data_dir + 'overunder_binary.bst')
def get_games(self, away_team, home_team):
    self.away_df = self.scoring_object.getTeamStats(away_team, self.game_date)
    self.home_df = self.scoring_object.getTeamStats(home_team, self.game_date)
    self.home_df.reset_index(drop=True, inplace=True)
    self.away_df.reset_index(drop=True, inplace=True)
    combined_df = pd.concat([self.home_df, self.away_df], axis=1)
    away_team = 'SAS'
    home_team = 'PHX'
    away_df = getTeamStats(away_team, pd.to_datetime('today'))
    home_df = getTeamStats(home_team, pd.to_datetime('today'))
    home_df.reset_index(drop=True, inplace=True)
    away_df.reset_index(drop=True, inplace=True)
    combined_df = pd.concat([home_df, away_df], axis=1)
    features_row = combined_df.unstack().to_frame().reset_index(drop=True).T
    barometer = np.arange(165, 243, 1)
    output = pd.DataFrame()
    # barometer = np.append(barometer,217.5)
    for i in barometer:
        new_features = pd.concat([pd.Series(i), features_row], axis=1)
        outrow = new_features
        output = pd.concat([output, outrow])
    outdf = pd.DataFrame(output)
    outdf.columns = range(0,outdf.shape[1])
    outdf.reset_index(drop=True,inplace=True)
    # over_under_val = self.bst_overunder.predict(xgb.DMatrix(outdf))
    over_under_val = bst_overunder.predict(xgb.DMatrix(outdf))
    out_list_df = pd.concat([outdf.iloc[:,0],pd.DataFrame(over_under_val)],axis=1)

    out_list = out_list_df

    return out_list

outlist_df = out_list
outlist_df.columns = ['x','y']
outlist_df.plot(kind='line', x = 'x', y = 'y')
plt.show()

outlist_df_lac = outlist_df