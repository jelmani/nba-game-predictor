import pandas as pd
import numpy as np
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder
from nba_api.stats.static import teams

class NBADataIngestor:
    """Fetch NBA data via nba-api and produce a clean games dataset.

    Steps: fetch → filter (no WNBA, G League, Summer League) → type fixes → returns DataFrame.
    """

    def __init__(self, start_date: str = "", end_date: str = ""):
        if (start_date == "") or (end_date == ""):
            self.get_all = True
        else:
            self.start_date = start_date
            self.end_date = end_date
    
    def _fetch_raw(self) -> pd.DataFrame:
        if self.get_all:
            all_games_df = leaguegamefinder.LeagueGameFinder().get_data_frames()[0]
        else:
            all_games_df = leaguegamefinder.LeagueGameFinder(date_from_nullable=self.start_date, date_to_nullable=self.end_date).get_data_frames()[0]
        return all_games_df
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get team IDs from NBA
        all_teams = teams.get_teams()
        nba_team_ids = [team["id"] for team in all_teams]
        # Filter out WNBA, G league etc
        nba_games_df = df[
            (df["TEAM_ID"].isin(nba_team_ids))]
        
        # Convert dates to datetime objects
        nba_games_df["GAME_DATE"] = pd.to_datetime(nba_games_df["GAME_DATE"])

        # Get rid of summer league and preseason
        gid = nba_games_df['GAME_ID'].astype(str)
        is_regular = gid.str.startswith('002')
        is_playoff = gid.str.startswith('003')
        is_play_in = gid.str.startswith('004')

        nba_games_df = nba_games_df[is_regular | is_playoff | is_play_in]

        nba_games_df = nba_games_df.sort_values(["GAME_DATE"])

        return nba_games_df
    
    def build(self) -> pd.DataFrame:
        return self._preprocess(self._fetch_raw())