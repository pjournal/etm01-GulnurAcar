# Data Preparation & Feature Extraction----
suppressWarnings(suppressMessages(require(data.table)))
suppressWarnings(suppressMessages(require(TunePareto)))
suppressWarnings(suppressMessages(require(glmnet)))
suppressWarnings(suppressMessages(require(caret)))
suppressWarnings(suppressMessages(require(xgboost)))
suppressWarnings(suppressMessages(require(nnet)))
source('_data_preprocessing.r')
source('_feature_extraction.r')
source('_performance_metrics.r')
source('_train_models.r')


testStart=as.Date('2018-08-16')
trainStart=as.Date('2012-07-15')
rem_miss_threshold=0.02 #parameter for removing bookmaker odds with missing ratio greater than this threshold
matches_data_path='df9b1196-e3cf-4cc7-9159-f236fe738215_matches.rds'
odd_details_data_path='df9b1196-e3cf-4cc7-9159-f236fe738215_odd_details.rds'

# read data
matches_raw=readRDS(matches_data_path)
odd_details_raw=readRDS(odd_details_data_path)

matches_raw=unique(matches_raw)
odd_details_data_path=unique(odd_details_data_path)

# preprocess matches
matches=matches_data_preprocessing(matches_raw)

# preprocess odd data
odd_details=details_data_preprocessing(odd_details_raw,matches)

# extract open and close odd type features from multiple bookmakers
features=extract_features.openclose(matches,odd_details,pMissThreshold=rem_miss_threshold,trainStart,testStart)

# divide data based on the provided dates 
train_features=features[Match_Date>=trainStart & Match_Date<testStart] 
test_features=features[Match_Date>=testStart] 
not_included_feature_indices=c(1:8)
outcomes = cbind(test_features$Result_Away,test_features$Result_Home,test_features$Result_Tie)

# XgBoosting RF----
predicted_prob = train_RF(train_features, test_features,not_included_feature_indices)
RPS_RF = RPS_matrix(predicted_prob,outcomes)
Results=data.table(Algorithm = 'Random Forest',Result = mean(RPS_RF))

# Ordinal Logistic Regression----
# Away<Tie<Home
predicted_prob_OLM1 = train_OrdinalLog(train_features, test_features,not_included_feature_indices, leveling=c("Away", "Tie", "Home"), orderFlag=TRUE)
RPS_OLM1 = RPS_matrix(predicted_prob_OLM1,outcomes)
Results=rbind(Results,data.table(Algorithm = 'Ordinal Logistic Regression (Away<Tie<Home)',Result = mean(RPS_OLM1, na.rm = TRUE)))

# Home<Tie<Away
predicted_prob_OLM2 = train_OrdinalLog(train_features, test_features,not_included_feature_indices, leveling=c("Home", "Tie", "Away"), orderFlag=TRUE)
RPS_OLM2 = RPS_matrix(predicted_prob_OLM2,outcomes)
Results=rbind(Results,data.table(Algorithm = 'Ordinal Logistic Regression (Home<Tie<Away)',Result = mean(RPS_OLM2, na.rm = TRUE)))

  
# Multinomial Logistic Regression----
predicted_prob_MLR = train_Multinomial(train_features, test_features,not_included_feature_indices, leveling=c("Home", "Tie", "Away"))
RPS_MLR = RPS_matrix(predicted_prob_MLR,outcomes)
Results=rbind(Results,data.table(Algorithm = 'Multinomial Logistic Regression',Result = mean(RPS_MLR, na.rm = TRUE)))


# Multinomial Logistic Regression with lasso penalty----
result_glmlasso = train_glmnet(train_features,test_features,not_included_feature_indices)
RPS = RPS_matrix(result_glmlasso[,3:5],outcomes)
Results=rbind(Results,data.table(Algorithm = 'Multinomial Logistic Regression with Lasso Penalty',Result = mean(RPS, na.rm = TRUE)))

Results