#' ---
#' title: "Machine Learning and Human Activity Recognition: Weight Lifting Exercises Dataset"
#' author: "@jrcajide"
#' date: 07-08-2016
#' output: 
#'  html_document
#' ---

#+ include=FALSE
rm(list = ls());gc(reset = T)
set.seed(1973)
knitr::opts_chunk$set(cache=T,fig.align='center',message=F,warning=F,echo = FALSE, results = 'asis',fig.width=10, fig.height=5)


# loading libraries -------------------------------------------------------
library(data.table)
library(knitr)
library(randomForest)
library(doParallel)
library(caret)
library(ggplot2)
library(ggthemes)
library(viridis)

# parallelizing -----------------------------------------------------------

numCores <- detectCores()
cl <- makeCluster(numCores - 2) 
registerDoParallel(cl) 

# importing data ----------------------------------------------------------

DT.train <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", stringsAsFactors = F, drop = 'V1', na.strings = c('','#DIV/0!','NA'))
DT.test <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", stringsAsFactors = F, drop = 'V1', na.strings = c('','#DIV/0!','NA'))

knitr::kable(head(DT.train))




barplot(table(DT.train$classe),col=viridis(5), border = "white")



# missing values ----------------------------------------------------------

sapply(DT.train, function(x) sum(is.na(x)) / nrow(DT.train) )
nas <- quantile(sapply(DT.train, function(x) sum(is.na(x))) / nrow(DT.train) , names = F)
nas[3]
DT.train <- 
  dim(DT.train[,sapply(DT.train, function(x) sum(is.na(x))) / nrow(DT.train) >= 0.9793089, with=FALSE])

# DT.train[ , lapply(.SD, function(x) sum(is.na(x)))]
DT.train[ , lapply(.SD, function(x) sum(is.na(x)) / nrow(DT.train))]


DT.train <- DT.train[, .SD, .SDcols=sapply(DT.train, function(x) (sum(is.na(x))) / nrow(DT.train)) < 0.9793089  ]
DT.test <- DT.test[, .SD, .SDcols=sapply(DT.test, function(x) (sum(is.na(x))) / nrow(DT.test)) < 0.9793089  ]


DT.train <- DT.train[, grep("classe|belt|arm|dumbbell",names(DT.train)), with=F]
# DT.test <- DT.test[, grep("classe|belt|arm|dumbbell",names(DT.test)), with=F]


DT.train <- DT.train[, which((names(DT.train) %in% names(DT.test)) | names(DT.train)=="classe"), with=F]

DT.train[, classe := as.factor(classe)]

# Cross validation
inTrain <- createDataPartition(y = DT.train$classe, p = 0.6, list = FALSE)
validationdata <- DT.train[-inTrain, ]
traindata <- DT.train[inTrain, ] 



head(traindata)

# Manual RF
bestmtry <- tuneRF(traindata[,-ncol(traindata), with=F],traindata$classe, ntreeTry=100, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)

rf <-randomForest(classe ~ . , data=traindata, mtry=5, ntree=1000, keep.forest=TRUE, importance=TRUE,test=validationdata)
rf


# Show model error
plot(rf, main = "Accuracy as a function of predictors", col=viridis(6))
legend('topright', colnames(rf$err.rate), col=viridis(6), fill=viridis(6))
# The black line shows the overall error rate which falls below 20%. The red and green lines show the error rate for ‘died’ and ‘survived’ respectively. 

# 4.3 Variable importance
# Let’s look at relative variable importance by plotting the mean decrease in Gini calculated across all trees.
# Get importance
importance    <- importance(rf)
varImportance <- data.frame(variables = row.names(importance), 
                            importance = round(importance[ ,'MeanDecreaseGini'],2))
library(dplyr)
# Create a rank variable based on importance
varImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(importance)))) %>% 
  arrange(desc(importance))

# CON DT
varImportance <- data.table(variables = row.names(importance), 
                            importance = round(importance[ ,'MeanDecreaseGini'],2))

varImportance[, Rank := min_rank(desc(importance))][order((Rank)),]

# Use ggplot2 to visualize the relative importance of variables
ggplot(varImportance, aes(x = reorder(variables, importance), y = importance, fill = importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = variables, y = 0.5, label = Rank), hjust=0, vjust=0.55, size = 3, colour = 'white') +
  labs(x = 'Variables') +
  scale_fill_viridis(discrete=F) +
  coord_flip() + 
  theme_few() +
  ggtitle("Relative importance of variables") + 
  theme(plot.title = element_text(lineheight=.8, face="bold"))


#PREDICT

# Predict using the test set
prediction <- predict(rf, DT.test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
# solution <- data.frame(user_name = DT.test$user_name, Survived = prediction)
DT.test$classe <- prediction

# Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes

barplot(table(DT.test$classe),col=viridis(5), border = "white", main="Labels assigned by the model", sub="Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes")

# As a conclusion only 35% of the exercises were done as specified by the participants

# Write the solution to file
fwrite(DT.test, "solution.csv", )
