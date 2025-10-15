import pandas as pd

class NBAFeatureExtractor:
    """Builds leak-free season-to-date features for NBA games."""

    def __init__(self, processed_games_df: pd.DataFrame) -> None:
        self.nba_games_df = processed_games_df
        self.nba_games_df["GAME_DATE"] = pd.to_datetime(self.nba_games_df["GAME_DATE"])
        self.stats = ['poss','ortg','drtg','eFG','tovr','orb%','ftr','pace', 'p_m']

        # Lazily built artifacts
        self.nba_games_joined = None
        self.games_by_home_w_stats = None
        self.games_by_away_w_stats = None
        self.all_games_w_stats = None
        self.feature_set = None

    # ---------- public API ----------

    def build(self) -> "NBAFeatureExtractor":
        """Run the full pipeline and return self for chaining."""
        self._join_and_calculate_stats()
        self._split_into_home_away_df()
        self._concat_home_and_away()
        self._calculate_season_to_date_stats_and_join()
        self._create_feature_df()
        return self

    def get_train(self, use_delta_stats: bool, train_end: str = "2023-01-01") -> tuple[pd.DataFrame, pd.Series]:
        """Return X_train, y_train using a time cutoff. Use about half of data for training set by default. Using delta stats minuses each teams corresponding 
        s2d stats results in slightly greater accuracy than leaving as is as well as reduces our total features in half"""
        self._require_features()
        training_set = self.feature_set[self.feature_set["GAME_DATE"] < pd.Timestamp(train_end)]
        final_x_train = self._make_x(training_set, use_delta_stats=use_delta_stats)
        final_y_train = training_set["WL"]
        return final_x_train, final_y_train

    def get_test(self, use_delta_stats: bool, start: str = "2023-01-01", end: str = "2024-01-01") -> tuple[pd.DataFrame, pd.Series]:
        """Return X_test, y_test for a date window."""
        self._require_features()
        test_set = self.feature_set[(self.feature_set["GAME_DATE"] >= pd.Timestamp(start)) & (self.feature_set["GAME_DATE"] < pd.Timestamp(end))]
        final_x_test = self._make_x(test_set, use_delta_stats=use_delta_stats)
        final_y_test = test_set["WL"]
        return final_x_test, final_y_test
    
    # ---------- pipeline steps ----------

    def _join_and_calculate_stats(self) -> None:
        self.nba_games_joined = self._combine_team_games(self.nba_games_df)
        self.nba_games_joined = self._append_calculated_stats_to_joined_df(self.nba_games_joined)

    def _split_into_home_away_df(self) -> None:
        a_subset = ["GAME_ID", "TEAM_ABBREVIATION_A"] + [f'{s}_A' for s in self.stats]
        nba_joined_subset_a = self.nba_games_joined[a_subset]
        self.games_by_home_w_stats = self.nba_games_df.merge(nba_joined_subset_a, left_on=["GAME_ID", "TEAM_ABBREVIATION"], right_on=["GAME_ID", "TEAM_ABBREVIATION_A"])

        b_subset = ["GAME_ID", "TEAM_ABBREVIATION_B"] + [f'{s}_B' for s in self.stats]
        nba_joined_subset_b = self.nba_games_joined[b_subset]
        self.games_by_away_w_stats = self.nba_games_df.merge(nba_joined_subset_b, left_on=["GAME_ID", "TEAM_ABBREVIATION"], right_on=["GAME_ID", "TEAM_ABBREVIATION_B"])

        # Redundant columns
        self.games_by_home_w_stats = self.games_by_home_w_stats.drop(columns=["TEAM_ABBREVIATION_A"])
        self.games_by_away_w_stats = self.games_by_away_w_stats.drop(columns=["TEAM_ABBREVIATION_B"])

    def _concat_home_and_away(self) -> None:
        # Home and away dataframes must have same column names for concat
        a_to_rename = [f'{s}_A' for s in self.stats]
        b_to_rename = [f'{s}_B' for s in self.stats]
        
        self.games_by_home_w_stats = self.games_by_home_w_stats.rename(columns=dict(zip(a_to_rename, self.stats)))
        self.games_by_away_w_stats = self.games_by_away_w_stats.rename(columns=dict(zip(b_to_rename, self.stats)))

        # Perform concat
        self.all_games_w_stats = pd.concat([self.games_by_home_w_stats, self.games_by_away_w_stats])

        self.all_games_w_stats['TEAM_ID'] = self.all_games_w_stats['TEAM_ID'].astype('int64')

    def _calculate_season_to_date_stats_and_join(self) -> None:
        all_games_copy = self.all_games_w_stats.sort_values(['TEAM_ID','SEASON_ID','GAME_DATE']).copy()
        all_games_copy['TEAM_ID'] = all_games_copy['TEAM_ID'].astype('int64')
        # Use groupby transform to create expanding season to date averages for each of the relevant stats for each team
        for s in self.stats:
            all_games_copy[f'{s}_S2D'] = (all_games_copy.groupby(['TEAM_ID','SEASON_ID'], sort=False)[s].transform(lambda x: x.shift(1).expanding().mean()))
        # Reduce to relevant columns
        reduced_all_games_copy = all_games_copy[["GAME_ID", "TEAM_ABBREVIATION", "MATCHUP"] + [f'{s}_S2D' for s in self.stats]]

        # Merge home game s2d stats. Merging reduced all games into home by mathcup ensures we merge the home games only
        self.games_joined_w_s2d_stats = self.games_by_home_w_stats.merge(reduced_all_games_copy, how="inner", on=["GAME_ID", "TEAM_ABBREVIATION", "MATCHUP"])

        # Get away games from all games df with s2d stats
        away_games_w_s2d = reduced_all_games_copy[(reduced_all_games_copy["MATCHUP"].str.contains("@"))]

        # Finally merge in away games with s2d to final joined df, suffixes differentiate between home (A) and away (B)
        self.games_joined_w_s2d_stats = self.games_joined_w_s2d_stats.merge(away_games_w_s2d, on="GAME_ID", suffixes=("_A", "_B"))


    def _create_feature_df(self) -> None:
        # The feature df is the last step before splitting into training/validation/test sets. It contains all of our features, our target y column (WL) and retains relevant info for identifying the games (date, team id etc) that can be used for splitting the data
        self.feature_set = self.games_joined_w_s2d_stats[["SEASON_ID", "GAME_DATE", "GAME_ID", "TEAM_NAME", "TEAM_ABBREVIATION_A", "TEAM_ABBREVIATION_B", "MATCHUP_A", "MATCHUP_B", "WL"] + [f'{s}_S2D_A' for s in self.stats] + [f'{s}_S2D_B' for s in self.stats]]
        self.feature_set = self.feature_set.dropna()
        self.feature_set['WL'] = (self.feature_set['WL'] == 'W').astype(int)

    # ---------- helpers ----------

    def _require_features(self) -> None:
        if self.feature_set is None:
            raise RuntimeError("Call .build() before requesting train/test sets.")
    
    def _make_x(self, frame: pd.DataFrame, use_delta_stats: bool = True) -> pd.DataFrame:
        if use_delta_stats:
            # Rename B columns so a/b has identical column names so we can minus them
            X = frame[[f'{s}_S2D_A' for s in self.stats]].subtract(frame[[f'{s}_S2D_B' for s in self.stats]].rename(columns=dict(zip([f'{s}_S2D_B' for s in self.stats], [f'{s}_S2D_A' for s in self.stats]))), fill_value=0)
            X.columns = [f'DELTA_{s}_S2D' for s in self.stats]
            return X
        else:
            return frame[[f'{s}_S2D_A' for s in self.stats] + [f'{s}_S2D_B' for s in self.stats]]


    # ---------- stat calculators ----------

    # Function from the NBA docs. Combine games into 1 row so 1 row per game
    @staticmethod
    def _combine_team_games(df: pd.DataFrame, keep_method: str ='home') -> pd.DataFrame:
        '''Combine a TEAM_ID-GAME_ID unique table into rows by game. Slow.
    
            Parameters
            ----------
            df : Input DataFrame.
            keep_method : {'home', 'away', 'winner', 'loser', ``None``}, default 'home'
                - 'home' : Keep rows where TEAM_A is the home team.
                - 'away' : Keep rows where TEAM_A is the away team.
                - 'winner' : Keep rows where TEAM_A is the losing team.
                - 'loser' : Keep rows where TEAM_A is the winning team.
                - ``None`` : Keep all rows. Will result in an output DataFrame the same
                    length as the input DataFrame.
                    
            Returns
            -------
            result : DataFrame
        '''
        # Join every row to all others with the same game ID.
        joined = pd.merge(df, df, suffixes=['_A', '_B'],
                          on=['SEASON_ID', 'GAME_ID', 'GAME_DATE'])
        # Filter out any row that is joined to itself.
        result = joined[joined.TEAM_ID_A != joined.TEAM_ID_B]
        # Take action based on the keep_method flag.
        if keep_method is None:
            # Return all the rows.
            pass
        elif keep_method.lower() == 'home':
            # Keep rows where TEAM_A is the home team.
            result = result[result.MATCHUP_A.str.contains(' vs. ')]
        elif keep_method.lower() == 'away':
            # Keep rows where TEAM_A is the away team.
            result = result[result.MATCHUP_A.str.contains(' @ ')]
        elif keep_method.lower() == 'winner':
            result = result[result.WL_A == 'W']
        elif keep_method.lower() == 'loser':
            result = result[result.WL_A == 'L']
        else:
            raise ValueError(f'Invalid keep_method: {keep_method}')
        return result

    @staticmethod
    def _append_calculated_stats_to_joined_df(joined_df: pd.DataFrame) -> pd.DataFrame:
        df = joined_df.copy()
        df['poss_A'] = df['FGA_A'] + 0.44*df['FTA_A'] + df['TOV_A'] - df['OREB_A']
        df['poss_B'] = df['FGA_B'] + 0.44*df['FTA_B'] + df['TOV_B'] - df['OREB_B']
        
        df['ortg_A'] = 100 * df['PTS_A'] / df['poss_A']
        df['ortg_B'] = 100 * df['PTS_B'] / df['poss_B']
        
        df['drtg_A'] = 100 * df['PTS_B'] / df['poss_B']
        df['drtg_B'] = 100 * df['PTS_A'] / df['poss_A']
        
        df['eFG_A']  = (df['FGM_A'] + 0.5*df['FG3M_A']) / df['FGA_A']
        df['eFG_B']  = (df['FGM_B'] + 0.5*df['FG3M_B']) / df['FGA_B']
        
        df['tovr_A'] = df['TOV_A'] / df['poss_A']
        df['tovr_B'] = df['TOV_B'] / df['poss_B']
        
        
        df['orb%_A'] = df['OREB_A'] / (df['OREB_A'] + df['DREB_B'])
        df['orb%_B'] = df['OREB_B'] / (df['OREB_B'] + df['DREB_A'])
        
        df['ftr_A']  = df['FTA_A'] / df['FGA_A']
        df['ftr_B']  = df['FTA_B'] / df['FGA_B']
        
        df['pace_A'] = 48 * (df['poss_A'] / (df['MIN_A']/5))
        df['pace_B'] = 48 * (df['poss_B'] / (df['MIN_B']/5))

        df['p_m_A'] = df["PLUS_MINUS_A"]
        df['p_m_B'] = df["PLUS_MINUS_B"]
        return df