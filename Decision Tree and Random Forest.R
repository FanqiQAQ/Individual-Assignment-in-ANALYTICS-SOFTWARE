# 加载必要的包
library(rpart)
library(rpart.plot)
library(caret)
library(ROCR)
library(randomForest)
library(gbm)
library(pROC)
library(themis)
library(ggplot2)

setwd("C:/Users/mengfanqi/Desktop/Analytics Software/data sets")
credit <- read.csv("credit.csv", stringsAsFactors = TRUE)
str(credit)

ggplot(credit, aes(x = default, fill = "yes")) +
  geom_bar() +
  labs(title = "distribution", x = "default", y = "num") +
  theme_minimal()

feature_plots <- list()
for (col in names(credit)[sapply(credit, is.numeric)]) {
  if (col != "yes") {
    feature_plots[[col]] <- ggplot(credit, aes_string(x = col, fill = "default")) +
      geom_density(alpha = 0.5) +
      labs(title = paste(col, "distribution")) +
      theme_minimal()
  }
}

# 2. 数据探索
cat("数据集结构:\n")
str(credit)

cat("\n数据摘要:\n")
summary(credit)
print(sapply(credit, function(x) sum(is.na(x))))

# 特征工程：创建新特征
credit$credit_age_ratio <- credit$other_credit / credit$age
# 3. 划分训练集和测试集
set.seed(9829)
train_sample <- sample(1000, 900)

str(train_sample)

# split the data frames
credit_train <- credit[train_sample, ]
credit_test  <- credit[-train_sample, ]

# 4. 构建决策树模型
#if neccessary
if (max(prop.table(table(credit_train$default))) > 0.7) {
  cat("\n检测到类别不平衡，使用SMOTE平衡数据...\n")
  credit_train <- SMOTE(default ~ ., data = credit_train, perc.over = 200, perc.under = 150)
  cat("\n平衡后训练集批准分布:\n")
  print(table(credit_train$default))
}

ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# 设置参数网格
tune_grid <- expand.grid(
  cp = seq(0.001, 0.02, length.out = 10)
)


# 训练模型
tree_model <- train(
  default ~ .,
  data = credit_train,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)

# 显示最佳参数
cat("\n最佳复杂度参数 (cp):", tree_model$bestTune$cp, "\n")

## Making some mistakes more costly than others
credit_boost10 <- C5.0(default ~ ., data = credit_train,
                       trials = 10)
credit_boost10
summary(credit_boost10)

credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# create dimensions for a cost matrix
matrix_dimensions <- list(c("no", "yes"))
names(matrix_dimensions) <- c("predicted", "actual")
matrix_dimensions

# build the matrix
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost

final_tree <- rpart(
  default ~ .,
  data = credit_train,
  method = "class",
  parms = list(split = "information"),
  control = rpart.control(
    minsplit = 10,
    minbucket = 5,
    cp = tree_model$bestTune$cp,
    maxdepth = 10
  )
)

# 5. 显示模型摘要
cat("\n决策树模型摘要:\n")
print(tree_model)

# 6. 可视化决策树
prp(final_tree, 
    extra = 104,        # 显示节点数和百分比
    box.palette = "GnBu", # 颜色方案
    branch.type = 5,    # 分支样式
    shadow.col = "gray", # 阴影颜色
    nn = TRUE)          # 显示节点编号

# 7. 变量重要性
cat("\n变量重要性:\n")
print(final_tree$variable.importance)

# 8. 在测试集上进行预测
predictions <- predict(final_tree, credit_test, type = "class")
prob_predictions <- predict(final_tree, credit_test, type = "prob")

# 9. 模型评估
confusion_matrix <- confusionMatrix(predictions, credit_test$default)
cat("\n混淆矩阵:\n")
print(confusion_matrix)

# 10. 绘制ROC曲线
pred <- prediction(prob_predictions[,2], credit_test$default == "yes")
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE, main = "ROC曲线")
abline(a = 0, b = 1, lty = 2, col = "gray")
auc <- performance(pred, "auc")@y.values[[1]]
legend("bottomright", legend = paste("AUC =", round(auc, 3)), bty = "n")

# 11. 分析结果
cat("\n模型准确率:", confusion_matrix$overall['Accuracy'], "\n")
cat("AUC值:", auc, "\n")


rf_model <- randomForest(
  default ~ .,
  data = credit_train,
  ntree = 200,
  importance = TRUE
)

rf_predictions <- predict(rf_model, credit)
rf_confusion <- confusionMatrix(rf_predictions, credit$default)

# 梯度提升机
gbm_model <- gbm(
  as.numeric(default == "yes") ~ .,
  data = credit_train,
  distribution = "bernoulli",
  n.trees = 200,
  interaction.depth = 3,
  shrinkage = 0.1
)

gbm_probs <- predict(gbm_model, credit, n.trees = 200, type = "response")
gbm_predictions <- ifelse(gbm_probs > 0.5, "yes", "no")
gbm_predictions <- factor(gbm_predictions, levels = levels(credit$default))
gbm_confusion <- confusionMatrix(gbm_predictions, credit$default)

# 12. 比较模型性能
cat("\n模型性能比较:\n")
cat("决策树准确率:", round(confusion_matrix$overall['Accuracy'], 3), "\n")
cat("随机森林准确率:", round(rf_confusion$overall['Accuracy'], 3), "\n")
cat("梯度提升机准确率:", round(gbm_confusion$overall['Accuracy'], 3), "\n")

# 13. 特征重要性分析（随机森林）
varImpPlot(rf_model, main = "随机森林特征重要性")

best_model <- ifelse(which.max(c(
  confusion_matrix$overall['Accuracy'],
  rf_confusion$overall['Accuracy'],
  gbm_confusion$overall['Accuracy']
)) == 1, "决策树",
ifelse(which.max(c(
  confusion_matrix$overall['Accuracy'],
  rf_confusion$overall['Accuracy'],
  gbm_confusion$overall['Accuracy']
)) == 2, "随机森林", "梯度提升机"))

cat("\n最佳模型:", best_model, "\n")
