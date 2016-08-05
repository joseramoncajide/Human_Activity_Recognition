
train.df <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
test.df <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv')

names(train.df)
table(train.df$classe)

# # Clase A = ok
# # Clase !A = nok
# 
# train.df$y = ifelse(train.df$classe == 'A' ,1,0)
# table(train.df$y)

library(randomForest)

head(train.df)
# get rid of columns where for ALL rows the value is NA
# train.df <- train.df[,colSums(is.na(train.df))<nrow(train.df)]
# train.df$classe <- NULL
ncol(train.df)

# vars

countnas <- function(data) {
  sum(is.na(data))
}

numbernas <- function(data) {
  apply(data, MARGIN = 2, countnas)
}

fractionnas <- function(data) {
  numbernas(data)/nrow(data)   
}

print(fractionnas(train.df)[which(fractionnas(train.df) > 0)])

cleandata <- function(data){
  data <- data[ , -which(fractionnas(data) > 0.97)]
  data[, grep("classe|belt|arm|dumbbell",names(data))]
}

train.df <- cleandata(train.df)
test.df <- cleandata(test.df)
tail(train.df)

train.df <- train.df[, which((names(train.df) %in% names(test.df)) | names(train.df)=="classe")]
library(caret)
inTrain <- createDataPartition(y = train.df$classe, p = 0.6, list = FALSE)
validationdata <- train.df[-inTrain, ]
traindata <- train.df[inTrain, ] 

head(traindata)
ncol(traindata)
head(train.df[-53])
bestmtry <- tuneRF(traindata[-53],traindata$classe, ntreeTry=100, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
bestmtry[4]

rf <-randomForest(classe ~ . , data=traindata, mtry=bestmtry[4], ntree=1000, keep.forest=TRUE, importance=TRUE,test=validationdata)
rf
round(importance(rf), 2)

# 4.3 Variable importance
# Let’s look at relative variable importance by plotting the mean decrease in Gini calculated across all trees.
# Get importance
importance    <- importance(rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

library(dplyr)
# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
library(ggplot2)
library(ggthemes)
# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

# Show model error
plot(rf, main = "Accuracy as a function of predictors")
legend('topright', colnames(rf$err.rate), col=1:6, fill=1:6)
# The black line shows the overall error rate which falls below 20%. The red and green lines show the error rate for ‘died’ and ‘survived’ respectively. 

#caret
rfFit <- train(classe ~ ., traindata, method = "rf", trControl=trainControl(method = 'cv', number = 5), allowParallel = TRUE)

print(rfFit)
print(rfFit$finalModel)
