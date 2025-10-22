from nba_feature_extractor import NBAFeatureExtractor
from nba_data_ingestor import NBADataIngestor
from log_reg_modeler import LogRegModeler
import argparse



# X_test, y_test = extractor.get_test()

# modeler.test_model(X_test, y_test)
parser = argparse.ArgumentParser(
    description="Use logistic regression to predict the winner of an NBA game."
)

parser.add_argument(
    "--home", help="Home team as an abbreviation"
)
parser.add_argument(
    "--away", help="Away Team as an abbreviation"
)

args = parser.parse_args()
home = args.home
away = args.away

ingestor = NBADataIngestor()
extractor = NBAFeatureExtractor(ingestor.build()).build()

if (home is None or away is None):
    print(f"Welcome to the NBA game predictor! Please run the program with the home and away teams as an abbreviation from this list as command line arguments: {extractor.get_team_abbr_list()}")
else:
    modeler = LogRegModeler()
    X_train, y_train = extractor.get_train()
    modeler.fit(X_train, y_train)
    feature_predict = extractor.get_feature_from_team_abbr(home, away)
    print(f"{home} has a {modeler.predict(feature_predict.to_numpy())} chance of winning against {away}")