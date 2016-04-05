library(caret)

# Load the data
filename <- "iris.csv"

# Training data
dataset <- read.csv(filename, header = FALSE)

# set the column names in the dataset
colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p = 0.80, list = FALSE)

# select 20% of the data for validation
# Test data
validation <- dataset[-validation_index,]

# dimensions of dataset, get number of instances and attributes
dim(dataset)

# list types for each attribute
sapply(dataset, class)

# take a peek at the first 5 rows of the data
head(dataset)

# list the levels for the class
levels(dataset$Species)

# summarize attribute distributions
summary(dataset)

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for (i in 1:4) {
  boxplot(x[,i], main = names(iris)[i])
}

# barplot for class breakdown
plot(y)

# scatterplot matrix - iteraction between variables
featurePlot(x, y, plot="ellipse")

# density plots for each attribute by class value
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
featurePlot(x, y, plot = "density", scales = scales)

# run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

# Try different models

# a) Linear algorithms
set.seed(7)
fit.lda <- train(Species~., data = dataset, method = "lda", metric = metric, trControl = control)

# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

# d) Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# select best model
# summarize accuracy of models
results <- resamples(list(lda = fit.lda, cart = fit.cart, knn = fit.knn, svm = fit.svm, rf = fit.rf))
summary(results)

# compare accuracy of models using plots
dotplot(results)

# Make predictions

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)