Machine learning course project
Project introduction
Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
Data
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.
Assignment
The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.
Getting and Cleaning Data
Load library
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
Getting Data
training_data <- read.csv("pml-training.csv")
testing_data <- read.csv("pml-testing.csv")
inTrain <- createDataPartition(training_data$classe, p=0.6, list=FALSE)
myTraining <- training_data[inTrain, ]
myTesting <- training_data[-inTrain, ]
Cleaning Data
# remove variables with nearly zero variance
nzv <- nearZeroVar(myTraining)
myTraining <- myTraining[, -nzv]
myTesting <- myTesting[, -nzv]

# remove variables that are almostly NA
mostlyNA <- sapply(myTraining, function(x) mean(is.na(x))) > 0.95
myTrainig <- myTraining[, mostlyNA==F]
myTesting <- myTesting[, mostlyNA==F]

# remove identification only variables (columns 1 to 5)
myTraining <- myTrainig[, -(1:5)]
myTesting  <- myTesting[, -(1:5)]
Predict Data by various models
1. Random forest
modFit <- randomForest(classe ~ ., data=myTraining)
modFit
## 
## Call:
##  randomForest(formula = classe ~ ., data = myTraining) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.35%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    7 2271    1    0    0 0.0035103115
## C    0    7 2047    0    0 0.0034079844
## D    0    0   20 1910    0 0.0103626943
## E    0    0    0    5 2160 0.0023094688
# Prediction using Random forest
predict <- predict(modFit, myTesting, type="class")
confusionMatrix(myTesting$classe, predict)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    1 1512    5    0    0
##          C    0    6 1362    0    0
##          D    0    0   13 1272    1
##          E    0    0    0    4 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9945, 0.9974)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9960   0.9870   0.9969   0.9993
## Specificity            1.0000   0.9991   0.9991   0.9979   0.9994
## Pos Pred Value         1.0000   0.9960   0.9956   0.9891   0.9972
## Neg Pred Value         0.9998   0.9991   0.9972   0.9994   0.9998
## Prevalence             0.2846   0.1935   0.1759   0.1626   0.1834
## Detection Rate         0.2845   0.1927   0.1736   0.1621   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9998   0.9975   0.9930   0.9974   0.9993
2. Decision tree
modFit_T <- rpart(classe~., myTraining)

# Prediction using Decision tree
predict_T <- predict(modFit_T, myTesting, type="class")
confusionMatrix(myTesting$classe, predict_T)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2011  100   11   98   12
##          B  268  915   78  189   68
##          C   66   26 1180   84   12
##          D   84  123  164  868   47
##          E   89  213   88  199  853
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7427          
##                  95% CI : (0.7328, 0.7523)
##     No Information Rate : 0.3209          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6733          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7986   0.6645   0.7758   0.6036   0.8599
## Specificity            0.9585   0.9068   0.9703   0.9348   0.9141
## Pos Pred Value         0.9010   0.6028   0.8626   0.6750   0.5915
## Neg Pred Value         0.9097   0.9270   0.9474   0.9131   0.9783
## Prevalence             0.3209   0.1755   0.1939   0.1833   0.1264
## Detection Rate         0.2563   0.1166   0.1504   0.1106   0.1087
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8786   0.7856   0.8730   0.7692   0.8870
3. Generalized Boosted Model (GBM)
control_GBM <- trainControl(method = "repeatedcv", number=5, repeats=1)
modFit_GBM <- train(classe~., myTraining, method="gbm", trControl=control_GBM, verbose=FALSE)
# Prediction using GBM
predict_GBM <- predict(modFit_GBM, myTesting)
confusionMatrix(predict_GBM, myTesting$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227   15    0    0    1
##          B    5 1477   19    9    4
##          C    0   23 1348   18    2
##          D    0    3    1 1258    6
##          E    0    0    0    1 1429
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9864          
##                  95% CI : (0.9835, 0.9888)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9827          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9730   0.9854   0.9782   0.9910
## Specificity            0.9971   0.9942   0.9934   0.9985   0.9998
## Pos Pred Value         0.9929   0.9756   0.9691   0.9921   0.9993
## Neg Pred Value         0.9991   0.9935   0.9969   0.9957   0.9980
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1882   0.1718   0.1603   0.1821
## Detection Prevalence   0.2859   0.1930   0.1773   0.1616   0.1823
## Balanced Accuracy      0.9975   0.9836   0.9894   0.9884   0.9954
Error and Cross validation
Random forest, Dicision tree, and GBM models give us 99.6 %, 75.4 %, and 98.8 % as accuracy, respectively.
The expected sample errors for Random forest, Dicision tree, and GBM are 0.4 %, 24.6 %, and 1.2 %, respectively.
Final test
Run the algorithm to the 20 test cases in the test data using most accurate model Random forest.
predict_test <- predict(modFit, testing_data, type = "class")
predict_test
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
