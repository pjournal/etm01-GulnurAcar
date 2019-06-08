require(data.table)
require(TunePareto)
require(glmnet)
require(caret)
setwd('C:/Users/VYKM 2/Desktop/ETM58D/parameter_tuning_boosting/')

testStart=as.Date('2018-08-16')
trainStart=as.Date('2012-07-15')
rem_miss_threshold=0.02 #parameter for removing bookmaker odds with missing ratio greater than this threshold

source('data_preprocessing.r')
source('feature_extraction.r')
source('performance_metrics.r')
source('train_models.r')

matches_data_path='C:/Users/VYKM 2/Desktop/ETM58D/df9b1196-e3cf-4cc7-9159-f236fe738215_matches.rds'
odd_details_data_path='C:/Users/VYKM 2/Desktop/ETM58D/df9b1196-e3cf-4cc7-9159-f236fe738215_odd_details.rds'

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


not_included_feature_indices=c(1:5)

# tuning with boosting using XgBoost with the help of caret package
# example from https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
nrounds=500
tune_grid <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = c(0.0025, 0.005),
  max_depth = c(4, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 5,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 5, # with n folds 
  verboseIter = TRUE, # show training log
  allowParallel = FALSE # FALSE for reproducible results 
)

xgb_tune <- caret::train(
  x = train_features[,-not_included_feature_indices,with=F],
  y = train_features$Match_Result,
  trControl = tune_control,
  tuneGrid = tune_grid,
  objective = 'multi:softprob',
  method = "xgbTree",
  verbose = TRUE
)

# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$Accuracy, probs = probs), min(x$results$Accuracy))) +
    theme_bw()
}

tuneplot(xgb_tune)

xgb_tune$bestTune

final_grid <- expand.grid(
  nrounds = xgb_tune$bestTune$nrounds,
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune$bestTune$max_depth,
  gamma = xgb_tune$bestTune$gamma,
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  min_child_weight = xgb_tune$bestTune$min_child_weight,
  subsample = xgb_tune$bestTune$subsample
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE, # FALSE for reproducible results
  classProbs=TRUE,
)

xgb_model <- caret::train(x = train_features[,-not_included_feature_indices,with=F], y = train_features$Match_Result, 
                          trControl = train_control,
                          tuneGrid = final_grid,
                          method = "xgbTree",
                          verbose = TRUE,
                          objective = 'multi:softprob'
)

predicted=predict(xgb_model, newdata = test_features)

table(test_features$Match_Result,predicted)

predicted_prob=predict(xgb_model, newdata = test_features,type='prob')

test_results=data.table(test_features[,list(matchId,Match_Date,Match_Result,Odd_Close_odd1_Pinnacle,Odd_Close_oddX_Pinnacle,Odd_Close_odd2_Pinnacle)],
                        predicted=predicted,predicted_prob)

test_results[,profit:=ifelse(predicted!=Match_Result,-1,NA)]
test_results[is.na(profit),profit:=ifelse(Match_Result=='Tie',Odd_Close_oddX_Pinnacle-1,ifelse(Match_Result=='Away',Odd_Close_odd2_Pinnacle-1,Odd_Close_odd1_Pinnacle-1))]

# profit by result type
test_results[!is.na(Match_Result),list(sum_profit=sum(profit),match_count=.N),by=list(Match_Result)]

# profit by bet type
test_results[!is.na(Match_Result),list(sum_profit=sum(profit),match_count=.N),by=list(predicted)]


test_results= test_results[order(Match_Date)]
test_results[,cumul_profit:=cumsum(profit)]
plot(test_results[,list(Match_Date,cumul_profit)])


rf_grid = expand.grid(mtry = c(3,10,20,40),
                      splitrule = c("gini", "extratrees"), 
                      min.node.size = c(1, 3, 5))

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 5, # with n folds 
  verboseIter = TRUE, # show training log
  allowParallel = FALSE # FALSE for reproducible results 
)

com_train= train_features[complete.cases(train_features)]
ranger_model <- caret:: train(x = com_train [,-not_included_feature_indices,with=F], y = com_train$Match_Result, 
                              trControl = tune_control,
                              tuneGrid = rf_grid,
                              method = "ranger",
                              verbose = TRUE
)
tuneplot(ranger_model)

ranger_model$bestTune


test_features= test_features[complete.cases(test_features)]
predicted=predict(ranger_model, newdata = test_features)

table(test_features$Match_Result,predicted)


test_results=data.table(test_features[,list(matchId,Match_Date,Match_Result,Odd_Close_odd1_Pinnacle,Odd_Close_oddX_Pinnacle,Odd_Close_odd2_Pinnacle)],
                        predicted=predicted)

test_results[,profit:=ifelse(predicted!=Match_Result,-1,NA)]
test_results[is.na(profit),profit:=ifelse(Match_Result=='Tie',Odd_Close_oddX_Pinnacle-1,ifelse(Match_Result=='Away',Odd_Close_odd2_Pinnacle-1,Odd_Close_odd1_Pinnacle-1))]

# profit by result type
test_results[!is.na(Match_Result),list(sum_profit=sum(profit),match_count=.N),by=list(Match_Result)]

# profit by bet type
test_results[!is.na(Match_Result),list(sum_profit=sum(profit),match_count=.N),by=list(predicted)]
test_results= test_results[order(Match_Date)]
test_results[,cumul_profit:=cumsum(profit)]
plot(test_results[,list(Match_Date,cumul_profit)])


dat1=matrix(runif(200,min=2.4,max=3),ncol=2)
dat2=matrix(runif(200,min=1.5,max=2.5),ncol=2)
cls=c(rep(1,100),rep(2,100))
full_dat=rbind(dat1,dat2)

plot(full_dat[,1],full_dat[,2],col=cls)

nof_noise=50
noise_binary=matrix(as.numeric(runif(200*nof_noise)>0.5),ncol=nof_noise)
noisy_dat=cbind(full_dat,noise_binary)

require(randomForest)

unsRF=randomForest(noisy_dat)

dissim=sqrt(1-unsRF$proximity)

transformed=cmdscale(dissim)

par(mfrow=c(2,1))

plot(full_dat,col=cls)

plot(transformed,col=cls)


scaled_data=scale(noisy_dat)
scaled_dist=dist(scaled_data)
transformed_scaled=cmdscale(scaled_dist)

par(mfrow=c(1,3))

plot(full_dat,col=cls)
plot(transformed,col=cls)
plot(transformed_scaled,col=cls)