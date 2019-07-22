library(tidyverse)
library(caret)
library(doParallel)
library(ROCR)

#AUCを算出するための関数
calAUC <- function(predictions, labels){
  pred <- prediction(predictions, labels)
  auc.tmp <- performance(pred,"auc")
  auc <- as.numeric(auc.tmp@y.values)
  message(paste0("> AUC:",auc))
  
  return(auc)
}

# selective lavelsのデータセットのXBG学習を行う関数
trainXGB_withSSL <- function(bank_train_sl, alloc_pred, Threshold = 0.9){
  bank_train_with_semiLabels <- bank_train_sl %>% 
    mutate(alloc_pred = alloc_pred$`TRUE`,
           R = alloc_pred > Threshold,
           argment = alloc_pred < 0.05,
           y = y %>% as.integer(.)-1) %>% 
    filter(bb == TRUE | argment == TRUE) %>% 
    mutate(y_tmp = if_else(bb == FALSE, -1, y)) %>% 
    mutate(label_Ds = if_else(R, as.integer(bb) - 1, 0),
           label_Ys = if_else(R, as.character(y_tmp), "0")) 
  
  #print(dim(bank_train_with_semiLabels))
  # 最終的なテスト
  model <- train(label_Ys ~ .,
                 data = bank_train_with_semiLabels %>% select(1:16, label_Ys), 
                 method = "xgbTree", 
                 metric = "Kappa",
                 tunelength = 4,
                 preProcess = c("center", "scale"),
                 trControl = ctrl)
  return(model)
}

#並列処理、使用するコア数4(使用できるコア数を知りたい場合は「detectCores()」を実行すればわかる)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

ctrl <- trainControl(method = "cv",
                     number = 4,
                     selectionFunction = "best")

#データ読み込み
bank_df_org <- read.csv("data/bank-full.csv")
bank_df <- bank_df_org %>% 
  mutate(y = as.factor(y)) %>% 
  sample_frac(0.2)

# 10分割
N <- 10
holdout <- split(sample(1:nrow(bank_df)), 1:N)

all_auc <- c()
all_samplen <- c()

for(i in 1:1){
  message(paste0("####   N=",i,"   ####"))
  
  auc <- c()
  samplen <- c()
  
  bank_test <- bank_df %>% 
    slice(holdout[[i]])
  
  bank_train <- bank_df %>% 
    anti_join(bank_test, key = "id")
  
  bank_pre_train <- bank_train %>% 
    sample_frac(0.4)
  
  bank_train <- bank_train %>% 
    anti_join(bank_pre_train, key = "id") 
  
  # スクリーニングモデルを作成
  model_make_blackbox <- train(y ~ .,
                               data = bank_pre_train, 
                               method = "xgbTree", 
                               trControl = ctrl,
                               verbose = TRUE)
  
  # ブラックボックスの対応を表すラベルを生成
  pred_lavels <- predict(model_make_blackbox, bank_train, type = "prob") 
  print(table(pred_lavels$yes > 0.15, bank_train$y))
  select_labels <- pred_lavels$yes > 0.15
  bank_train_sl <- bank_train %>% 
    mutate(bb = as.factor(select_labels))

  #テストデータの予測精度を確認
  #selective labels状態のデータでの予測精度を確認
  bank_sl.train <- bank_train_sl %>% 
    filter(bb == TRUE) %>% 
    select(-bb) %>% 
    mutate(y = as.factor(y))
  
  model_xgb_tune <- train(y ~ .,
                          data = bank_sl.train, 
                          method = "xgbTree", 
                          metric = "Kappa",
                          tunelength = 4,
                          preProcess = c("center", "scale"),
                          trControl = ctrl,
                          verbose = TRUE)
  
  pred <- predict(model_xgb_tune, newdata = bank_test, type = "prob")
  auc[1] <- calAUC(pred$yes,bank_test$y)
  samplen[1] <- nrow(bank_sl.train)

  # selective labelなしの場合
  bank_sl.train_all <- bank_train_sl %>% 
    select(-bb) %>% 
    mutate(y = as.factor(y))
  
  model_xgb_tune2 <- train(y ~ .,
                           data = bank_sl.train_all, 
                           method = "xgbTree", 
                           metric = "Kappa",
                           tunelength = 4,
                           preProcess = c("center", "scale"),
                           trControl = ctrl)
  
  pred2 <- predict(model_xgb_tune2, newdata = bank_test, type = "prob")
  auc[2] <- calAUC(pred2$yes, bank_test$y)
  samplen[2] <- nrow(bank_sl.train)
  
  #割当てのモデルをランダムフォレストで作成
  allocation_model <- train(bb ~ .,
                            data = bank_train_sl[,-c(17)], 
                            method = "rf", 
                            trControl = ctrl,
                            verbose = TRUE
  )
  
  alloc_pred <- predict(allocation_model, newdata = bank_train_sl[,-c(17)], type = "prob")
  hist(alloc_pred$`TRUE`)
  
  # 最終的なテスト
  model_test <- trainXGB_withSSL(bank_train_sl, alloc_pred, Threshold = 0.9)
  pred_lst <- predict(model_test, newdata = bank_test, type = "prob")
  auc[3] <- calAUC(pred_lst$`1`,bank_test$y)
  samplen[3] <- nrow(model_test$trainingData)

  model_test <- trainXGB_withSSL(bank_train_sl, alloc_pred, Threshold = 0.8)
  pred_lst <- predict(model_test, newdata = bank_test, type = "prob")
  auc[4] <- calAUC(pred_lst$`1`,bank_test$y)
  samplen[4] <- nrow(model_test$trainingData)
  
  model_test <- trainXGB_withSSL(bank_train_sl, alloc_pred, Threshold = 0.7)
  pred_lst <- predict(model_test, newdata = bank_test, type = "prob")
  auc[5] <- calAUC(pred_lst$`1`,bank_test$y)
  samplen[5] <- nrow(model_test$trainingData)
  
  all_auc <- rbind(all_auc, auc)
  all_samplen <- rbind(all_samplen, samplen)
}

stopCluster(cl)

write.csv(all_auc, "all_auc_bank.csv")
write.csv(all_samplen, "all_samplen_bank.csv")


