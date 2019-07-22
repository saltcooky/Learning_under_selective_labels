#グラフ作成

Threshold = 0.9

#(a)
#sl状態のデータでの出力、学習のフィット状況
# 得られてデータで学習した２モデルのテストデータでの出力
pred <- predict(model_xgb_tune, newdata = bank_test, type = "prob") 
alloc_pred_test <- predict(allocation_model, newdata = bank_test, type = "prob")
bb_pred_test <- predict(model_make_blackbox, newdata = bank_test, type = "prob")

brank_label <- bank_test %>% 
  mutate(bb = bb_pred_test$yes> 0.15) %>% 
  mutate(y = ifelse(bb == TRUE, y, "non"))

plot_df1 <- data.frame(pred_human = alloc_pred_test$`TRUE`,
                       pred_plcm = pred$yes,
                       labels = as.factor(brank_label$y))

ggplot(plot_df1, aes(x = pred_human, y = pred_plcm))+
  geom_point(aes(colour = labels, alpha=0.6))+
  labs(title="(a) Real-world, observed")+xlim(0,1)+ylim(0,1)

#(d)
# 水増し再ラベリングデータで学習した２モデルのテストデータでの出力
pred <- predict(model_test, newdata = bank_test, type = "prob") #ok
alloc_pred_test <- predict(allocation_model, newdata = bank_test, type = "prob")
bb_pred_test <- predict(model_make_blackbox, newdata = bank_test, type = "prob")

bank_train_sl <- bank_train %>% 
  mutate(bb = as.factor(select_labels))

bank_train_with_semiLabels <- bank_train_sl %>% 
  mutate(alloc_pred = alloc_pred$`TRUE`,
         R = alloc_pred > 0.7,
         argment = alloc_pred < 0.05,
         y = y %>% as.integer(.)-1) %>% 
  filter(bb == TRUE | argment == TRUE) %>% 
  mutate(y_tmp = if_else(bb == FALSE, -1, y)) %>% 
  mutate(label_Ds = if_else(R, as.integer(bb) - 1, 0),
         label_Ds = as.factor(label_Ds),
         label_Ys = if_else(R, as.character(y_tmp), "0")) 

#割当てのモデルをランダムフォレストで作成
allocation_model_new <- train(label_Ds ~ .,
                              data = bank_train_with_semiLabels %>% select(1:16,label_Ds), 
                              method = "rf", 
                              trControl = ctrl,
                              verbose = TRUE
)


alloc_pred_new <- predict(allocation_model, newdata = bank_test, type = "prob")

brank_label <- bank_test %>% 
  mutate(bb = bb_pred_test$yes> 0.15) %>% 
  mutate(y = ifelse(bb == TRUE, y, "non"))

plot_df2 <- data.frame(pred_human = alloc_pred_new[,2],
                       pred_plcm = pred$`1`,
                       labels = as.factor(brank_label$y))

ggplot(plot_df2, aes(x = pred_human, y = pred_plcm))+
  geom_point(aes(colour = labels, alpha=0.6))+
  labs(title="(d) Semisynthetic, augmented")+xlim(0,1)+ylim(0,1)


#(c)
# 水増しなし再ラベリングデータで学習した２モデルのテストデータでの出力
#縦軸：再ラベリングした学習データの最終モデルの出力結果
#横軸：再ラベリングした学習データで学習した割り当てモデルの出力結果
alloc_pred <- predict(allocation_model, newdata = bank_train_sl[,-c(17)], type = "prob")

bank_train_with_semiLabels <- bank_train_sl %>% 
  mutate(alloc_pred = alloc_pred$`TRUE`,
         R = alloc_pred > 0.9,
         y = y %>% as.integer(.)-1) %>% 
  #filter(bb == TRUE) %>% 
  mutate(y_tmp = if_else(bb == FALSE, -1, y)) %>% 
  mutate(label_Ds = if_else(R, as.integer(bb) - 1, 0),
         label_Ds = as.factor(label_Ds),
         label_Ys = if_else(R, as.character(y_tmp), "0"))

alloc_semi_model <- train(label_Ds ~ .,
                          data = bank_train_with_semiLabels %>% select(1:16, label_Ds), 
                          method = "rf", 
                          trControl = ctrl)

pred_yoko <- predict(alloc_semi_model, newdata = bank_test, type = "prob")

model <- train(label_Ys ~ .,
               data = bank_train_with_semiLabels %>% select(1:16, label_Ys), 
               method = "xgbTree", 
               metric = "Kappa",
               tunelength = 4,
               preProcess = c("center", "scale"),
               trControl = ctrl)

pred_tate <- predict(model, newdata = bank_test, type = "prob")

plot_dfc <- data.frame(pred_human = pred_yoko$`1`,
                       pred_plcm = pred_tate$`1`,
                       labels = as.factor(brank_label$y))

ggplot(plot_dfc, aes(x = pred_human, y = pred_plcm))+
  geom_point(aes(colour = labels, alpha=0.6))+
  labs(title="(c) Semisynthetic, obderved")
