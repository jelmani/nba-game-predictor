from nba_feature_extractor import NBAFeatureExtractor
from nba_data_ingestor import NBADataIngestor
from log_reg_modeler import LogRegModeler

ingestor = NBADataIngestor()
extractor = NBAFeatureExtractor(ingestor.build()).build()

modeler = LogRegModeler()

X_train, y_train = extractor.get_train()
modeler.fit(X_train, y_train)

X_test, y_test = extractor.get_test()

modeler.test_model(X_test, y_test)