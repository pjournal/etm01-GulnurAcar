train_RF <- function(train_features, test_features,not_included_feature_indices=c(1:8)){
  set.seed(1)
  nrounds=500
  tune_grid <- expand.grid(
    nrounds = seq(from = 50, to = nrounds, by = 50), #boosting parameter
    eta = c(0.0025, 0.005), # boosting parameter
    max_depth = c(4, 6), # Tree parameter
    gamma = 0, # class imbalance problemleri icin parametre
    colsample_bytree = 1, # agaclara randomness katmak icin feature'larin yuzde kacini kullanayim
    min_child_weight = 5, # kucuk sayilar daha iyi calisir data buyukse kullan
    subsample = 1 # random forestta rastgele data secerek train edioduk burda hepsini kullan dedik
  ) # depth eta ve nround la oyna
  
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
    classProbs=TRUE
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
  return(predicted_prob)
}

train_OrdinalLog <- function(train_features, test_features,not_included_feature_indices=c(1:8),leveling, orderFlag=TRUE){
  require(MASS)
  
  train_features_OLR = copy(train_features)
  test_features_OLR = copy(test_features)
  
  train_features_OLR$Match_Result <- factor(train_features_OLR$Match_Result, levels=leveling, ordered=orderFlag)
  temp =  train_features_OLR[,5:length(colnames(train_features_OLR))]
  temp[,c("Result_Home","Result_Away","Result_Tie"):=NULL]
  model <- polr(Match_Result~., data=temp)  
  predicted_prob <- predict(model, test_features_OLR[,-not_included_feature_indices,with=F], type="p")
  return(predicted_prob)
}

train_Multinomial <- function(train_features, test_features,not_included_feature_indices=c(1:8),leveling){

  train_features_MLR = copy(train_features)
  test_features_MLR = copy(test_features)
  train_features_MLR$Match_Result <- factor(train_features_MLR$Match_Result, levels=leveling, ordered=FALSE)
  
  temp =  train_features_MLR[,5:length(colnames(test_features_MLR))]
  temp[,c("Result_Home","Result_Away","Result_Tie"):=NULL]
  model_MLR = multinom(Match_Result~. ,data = temp)
  predicted_prob_MLR <- predict(model_MLR, test_features_MLR[,-not_included_feature_indices,with=F], type="p")
  return(predicted_prob_MLR)
}

train_glmnet <- function(train_features, test_features,not_included_feature_indices=c(1:8), alpha=1,nlambda=50, tune_lambda=TRUE,nofReplications=2,nFolds=10,trace=T){
  
  set.seed(1)
  # train_features_MLRlasso = copy(train_features)
  # test_features_MLRlasso = copy(test_features)
  # glmnet works with complete data
  glm_features=train_features[complete.cases(train_features)]
  train_class=glm_features$Match_Result
  glm_train_data=glm_features[,-not_included_feature_indices,with=F]
  glm_test_data=test_features[,-not_included_feature_indices,with=F]
  if(tune_lambda){
    # to set lambda parameter, cross-validation will be performed and lambda is selected based on RPS performance
    
    cvindices=generateCVRuns(train_class,nofReplications,nFolds,stratified=TRUE)
    
    # first get lambda sequence for all data
    glmnet_alldata = glmnet(as.matrix(glm_train_data), as.factor(train_class), family="multinomial", alpha = alpha, nlambda=nlambda)
    lambda_sequence = glmnet_alldata$lambda
    
    cvresult=vector('list',nofReplications*nFolds)
    iter=1
    for(i in 1:nofReplications) {
      thisReplication=cvindices[[i]]
      for(j in 1:nFolds){
        if(trace){
          cat(sprintf('Iteration %d: Fold %d of Replication %d\n',iter,j,i))
        }
        testindices=order(thisReplication[[j]])
        
        cvtrain=glm_train_data[-testindices]    
        cvtrainclass=train_class[-testindices]   
        cvtest=glm_train_data[testindices]
        cvtestclass=train_class[testindices] 
        
        inner_cv_glmnet_fit = glmnet(as.matrix(cvtrain),as.factor(cvtrainclass),family="multinomial", alpha = alpha,lambda=lambda_sequence)
        valid_pred = predict(inner_cv_glmnet_fit, as.matrix(cvtest), s = lambda_sequence, type = "response")
        
        #check order of predictions
        order_of_class=attr(valid_pred,'dimnames')[[2]]
        new_order=c(which(order_of_class=='Home'),which(order_of_class=='Tie'),which(order_of_class=='Away'))
        foldresult=rbindlist(lapply(c(1:length(lambda_sequence)),function(x) { data.table(repl=i,fold=j,lambda=lambda_sequence[x],valid_pred[,new_order,x],result=cvtestclass)}))
        cvresult[[iter]]=foldresult
        iter=iter+1
      }
    }
    
    cvresult=rbindlist(cvresult)
    
    # creating actual targets for rps calculations
    cvresult[,pred_id:=1:.N]
    outcome_for_rps=data.table::dcast(cvresult,pred_id~result,value.var='pred_id')
    outcome_for_rps[,pred_id:=NULL]
    outcome_for_rps[is.na(outcome_for_rps)]=0
    outcome_for_rps[outcome_for_rps>0]=1
    setcolorder(outcome_for_rps,c('Home','Tie','Away'))
    
    # calculate RPS
    overall_results=data.table(cvresult[,list(repl,fold,lambda)],RPS=RPS_matrix(cvresult[,list(Home,Tie,Away)],outcome_for_rps))
    
    # summarize performance for each lambda
    overall_results_summary=overall_results[,list(RPS=mean(RPS)),list(repl,fold,lambda)]
    
    # find best lambdas as in glmnet based on RPS
    overall_results_summary=overall_results_summary[,list(meanRPS=mean(RPS),sdRPS=sd(RPS)),list(lambda)]
    overall_results_summary[,RPS_mean_lb := meanRPS - sdRPS]
    overall_results_summary[,RPS_mean_ub := meanRPS + sdRPS]
    
    cv_lambda_min=overall_results_summary[which.min(meanRPS)]$lambda
    
    semin=overall_results_summary[lambda==cv_lambda_min]$RPS_mean_ub
    cv_lambda.1se=max(overall_results_summary[meanRPS<semin]$lambda)
    
    cvResultsSummary = list(lambda.min =cv_lambda_min, lambda.1se = cv_lambda.1se,
                            meanRPS_min=overall_results_summary[lambda==cv_lambda_min]$meanRPS,
                            meanRPS_1se=overall_results_summary[lambda==cv_lambda.1se]$meanRPS)
    
  }
  
  # fit final glmnet model with the lambda with minimum error
  final_glmnet_fit = glmnet(as.matrix(glm_train_data),as.factor(train_class),family="multinomial", alpha = alpha,lambda=cvResultsSummary$lambda.min)
  # obtain predictions
  predicted_probabilities=predict(final_glmnet_fit, as.matrix(glm_test_data), type = "response")
  
  #check order of predictions
  order_of_class=attr(predicted_probabilities,'dimnames')[[2]]
  new_order=c(which(order_of_class=='Away'),which(order_of_class=='Home'),which(order_of_class=='Tie'))
  
  final_result=data.table(test_features[,list(matchId,Match_Result)],predicted_probabilities[,new_order,1])
  
  return(final_result)
  # return(list(predictions=final_result,cv_stats=cvResultsSummary))
}