# Data Source: http://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
# Related Article: Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor.
  #Computational Materials Science,154, 346–354. doi: 10.1016/j.commatsci.2018.07.052



rm(list = ls())    #delete objects
cat("\014")
library(randomForest)
library(glmnet)
library(ggplot2)
library(dplyr)
set.seed(111)


####### Original Data #######
df_raw = read.csv('Superconductor.csv')
# View(df)
# dim(df_raw) # 21263 Variables * 81 features #Col82 is the target
# sum(is.na(df_raw)) # no missing value
# colMeans(df_raw) 



####### Data Engineering

#===Standardize Features===
# Per requirements of this project, we will standardize using Equation 6.6 on ISLR 
scale_ISLR = function(x){x/sd(x)}

# Column 1 is categorical data that describes # of elements in a superconductor
# which will not be used our regression
df = df_raw[-1] 
# dim(df) # 21263 Variables * 80 features #Col81 is the target

# standardize feactures col1-col80
for (i in 1:(ncol(df)-1)) {
  df[,i] = scale_ISLR(df[,i])
}

rm(i)

### There are 21,263 observations in total. However, for the sake of this project,
### in which we focuse on comparing Lasso, Ridge, Elastic-net and Random forest in 
### regression performance, I will select 1000 obs randomly
### (training Random Forest on the full data set was time consuming)
n_obs = 1000
df_index_shuffled = sample(dim(df)[1])
index_selected = df_index_shuffled[1:n_obs]
df = df[index_selected,]

####### Train-Test Split ######

n = nrow(df) # number of observations
p = ncol(df)-1 # number of features

y = df[,81] 
X = data.matrix(df[,-81])

ratio = 0.8 # 80% Train + 20% Test
n_train = floor(ratio * n)
n_test = n-n_train

####### Train and Predict ######
# Models used in this project: Lasso, Ridge, Elastic-Net (alpha = 0.5), Random Forest

R = 100 # repeat the train-test splits and model fitting 100 times -> 100 samples

# rsq.train/test.model: store all R^2 of 4 models in 100 samples
  # R^2 = 1-mean((y_train - y_train/test_hat)^2)/mean((y - mean(y))^2)
  # noted that the denominator is calculated based on whole dataset in order to compare R^2 of
  # train and test dataset.

# res.train/test.model: Matrix to store all residuals of 4 models in 100 samples
  # residuals of train/test = y_train/test_hat - y
  # e.g. Column_i would be residuals of the ith sample

rsq_train_lasso = rep(0,R) # Lasso (alpha = 1)
rsq_test_lasso = rep(0,R)
res_train_lasso = matrix(0,n_train,R)
res_test_lasso = matrix(0,n_test,R) 

rsq_train_ridge = rep(0,R) # Ridge (alpha = 0)
rsq_test_ridge = rep(0,R)
res_train_ridge= matrix(0,n_train,R)
res_test_ridge = matrix(0,n_test,R)

rsq_train_en = rep(0,R) # Elastic-Net (alpha = 0.5)
rsq_test_en = rep(0,R)
res_train_en = matrix(0,n_train,R)
res_test_en = matrix(0,n_test,R)

rsq_train_rf= rep(0,R) # Random Forest
rsq_test_rf = rep(0,R)
res_train_rf = matrix(0,n_train,R)
res_test_rf = matrix(0,n_test,R)

# one iteration: user time 6.244s
for (r in 1:R) {
    shuffled_indexes = sample(n)
    train = shuffled_indexes[1:n_train]
    test = shuffled_indexes[(1+n_train):n]
    X_train = X[train,]
    y_train = y[train]
    X_test = X[test,]
    y_test = y[test]
    
    # fit lasso (alpha = 1)
    lasso.cv.fit = cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10) # cross-validation to find lambda
    lasso.fit = glmnet(X_train, y_train,alpha = 1, lambda = lasso.cv.fit$lambda.min)
    y_train_hat = predict(lasso.fit, X_train)
    y_test_hat = predict(lasso.fit, X_test)
    rsq_train_lasso[r] = 1-mean((y_train - y_train_hat)^2)/mean((y - mean(y))^2)
    rsq_test_lasso[r] = 1-mean((y_test - y_test_hat)^2)/mean((y - mean(y))^2)
    res_train_lasso[,r] = y_train_hat - y_train
    res_test_lasso[,r] = y_test_hat - y_test

    # fit ridge (alpha = 0)
    ridge.cv.fit = cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10)
    ridge.fit = glmnet(X_train, y_train,alpha = 0, lambda = ridge.cv.fit$lambda.min)
    y_train_hat = predict(ridge.fit, X_train)
    y_test_hat = predict(ridge.fit, X_test)
    rsq_train_ridge[r] = 1-mean((y_train - y_train_hat)^2)/mean((y - mean(y))^2)
    rsq_test_ridge[r] = 1-mean((y_test - y_test_hat)^2)/mean((y - mean(y))^2)
    res_train_ridge[,r] = y_train_hat - y_train
    res_test_ridge[,r] = y_test_hat - y_test

    # fit elastic-net (alpha = 0.5)
    en.cv.fit = cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10)
    en.fit = glmnet(X_train, y_train,alpha = 0.5, lambda = en.cv.fit$lambda.min)
    y_train_hat = predict(en.fit, X_train)
    y_test_hat = predict(en.fit, X_test)
    rsq_train_en[r] = 1-mean((y_train - y_train_hat)^2)/mean((y - mean(y))^2)
    rsq_test_en[r] = 1-mean((y_test - y_test_hat)^2)/mean((y - mean(y))^2)
    res_train_en[,r] = y_train_hat - y_train
    res_test_en[,r] = y_test_hat - y_test

    # fit Random Forest with mtry = sqrt(p)
    rf = randomForest(X_train, y_train, mtry = sqrt(p), importance = TRUE)
    y_test_hat = predict(rf, X_train)
    y_test_hat = predict(rf, X_test)
    rsq_train_rf[r] = 1-mean((y_train - y_train_hat)^2)/mean((y - mean(y))^2)
    rsq_test_rf[r] = 1-mean((y_test - y_test_hat)^2)/mean((y - mean(y))^2)
    res_train_rf[,r] = y_train_hat - y_train
    res_test_rf[,r] = y_test_hat - y_test
    
    cat(sprintf("r=%3.f| rsq_test_lasso=%.2f, rsq_test_ridge=%.2f, rsq_test_en=%.2f, rsq_test_rf=%.2f| \n       rsq_train_lasso=%.2f, rsq_train_ridge=%.2f, rsq_train_en=%.2f,rsq_train_rf=%.2f| \n", 
                r, rsq_test_lasso[r], rsq_test_ridge[r], rsq_test_en[r], rsq_test_rf[r],
                rsq_train_lasso[r], rsq_train_ridge[r], rsq_train_en[r], rsq_train_rf[r]))
}


####### Get some plots of our models #######

# The side-by-side boxplots of R^2_test,R^2_train for each model

par(mfrow=c(1,2))
boxplot(rsq_test_lasso, xlab = "Lasso Test R_Square")
boxplot(rsq_train_lasso, xlab = "Lasso Train R_Square")

par(mfrow=c(1,2))
boxplot(rsq_test_ridge, xlab = "Ridge Test R_Square")
boxplot(rsq_train_ridge, xlab = "Ridge Train R_Square")

par(mfrow=c(1,2))
boxplot(rsq_test_en, xlab = "Elastic Net Test R_Square")
boxplot(rsq_train_en, xlab = "Elastic Net Train R_Square")

par(mfrow=c(1,2))
boxplot(rsq_test_rf, xlab = "Random Forest Test R_Square")
boxplot(rsq_train_rf, xlab = "Random Forest Train R_Square")

#====== For one of the 100 samples

# a 10-fold CV curves for lasso, ridge and elastic-net α = 0.5
# Plotted the last sample
par(mfrow=c(1,3))
plot(lasso.cv.fit,sub = "CV for Lasso")
plot(ridge.cv.fit,sub = "CV for Ridge")
plot(en.cv.fit,sub = "CV for Elastic-Net")

# Plot the side-by-side boxplots of train and test residuals 
nth_sample = R # plot the last sample

par(mfrow=c(1,2))
boxplot(res_train_lasso[,nth_sample], xlab = "Residuals of Lasso Train")
boxplot(res_test_lasso[,nth_sample], xlab = "Residuals of Lasso Test")

par(mfrow=c(1,2))
boxplot(res_train_ridge[,nth_sample], xlab = "Residuals of Ridge Train")
boxplot(res_test_ridge[,nth_sample], xlab = "Residuals of Ridge Test")

par(mfrow=c(1,2))
boxplot(res_train_en[,nth_sample], xlab = "Residuals of Elastic-Net Train")
boxplot(res_test_en[,nth_sample], xlab = "Residuals of Elastic-Net Test")

par(mfrow=c(1,2))
boxplot(res_train_rf[,nth_sample], xlab = "Residuals of Random Forest Train")
boxplot(res_test_rf[,nth_sample], xlab = "Residuals of Random Forest Test")



####### Bootstrap ######

bootstrap_samples = 100 # 100 samples

# store betas of each model in 100 samples
# the column_i would be betas of a model in ith sample
beta_lasso_bs = matrix(0, nrow = p, ncol = bootstrap_samples)
beta_ridge_bs = matrix(0, nrow = p, ncol = bootstrap_samples)
beta_en_bs = matrix(0, nrow = p, ncol = bootstrap_samples)
beta_rf_bs = matrix(0, nrow = p, ncol = bootstrap_samples)

# Bootstrap & Fit models
for (i in 1:bootstrap_samples) {
  bs_indexes = sample(n, replace = T) # Bootstrap indexes
  X_bs = X[bs_indexes,]
  y_bs = y[bs_indexes]
  
  # Lasso 
  # p.s: bs stands for bootstrap
  #      we need to name variables different variables in models in previous models
  lasso.bs.cv = cv.glmnet(X_bs, y_bs, alpha = 1, nfolds = 10)
  lasso.bs.fit = glmnet(X_bs, y_bs, intercept = FALSE, alpha = 1, lambda = lasso.bs.cv$lambda.min)
  beta_lasso_bs[,i] = as.vector(lasso.bs.fit$beta)
  
  # Ridge
  ridge.bs.cv = cv.glmnet(X_bs, y_bs, alpha = 0, nfolds = 10)
  ridge.bs.fit = glmnet(X_bs, y_bs, intercept = FALSE, alpha = 1, lambda = ridge.bs.cv$lambda.min)
  beta_ridge_bs[,i] = as.vector(ridge.bs.fit$beta)
  
  # Elastic Net (alpha = 0.5)
  en.bs.cv = cv.glmnet(X_bs, y_bs, alpha = 0.5, nfolds = 10)
  en.bs.fit = glmnet(X_bs, y_bs, intercept = FALSE, alpha = 1, lambda = en.bs.cv$lambda.min)
  beta_en_bs[,i] = as.vector(en.bs.fit$beta)
  
  # Random Forest
  rf.bs = randomForest(X_bs, y_bs, mtry = sqrt(p), importance = TRUE) # mtry: Number of variables randomly sampled as candidates at each split
  beta_rf_bs[,i] = as.vector(rf.bs$importance[,1]) # For Regression, the first column is the mean decrease in accuracy
                                                  # note that here beta_rf_bs is not "betas" of random forest
  cat(sprintf("Bootstrap Sample %3.f \n", i))
}


# Calculate bootstrapped standard errors of each coefficient
lasso_bs_sd = apply(beta_lasso_bs, 1, "sd")
ridge_bs_sd = apply(beta_ridge_bs, 1, "sd")
en_bs_sd = apply(beta_en_bs, 1, "sd")
rf_bs_sd = apply(beta_rf_bs, 1, "sd")

# Fit models to the whole dataset: model.w.fit
  # Then, save betaS to a data frame
lasso.w = cv.glmnet(X, y, alpha = 1, nfolds = 10)
lasso.w.fit = glmnet(X, y, alpha = 1, lambda = lasso.w$lambda.min)
betaS_lasso = data.frame(c(1:p), as.vector(lasso.w.fit$beta), 2*lasso_bs_sd)
colnames(betaS_lasso) = c("Feature", "Value", "Error")

ridge.w = cv.glmnet(X, y, alpha = 0, nfolds = 10)
ridge.w.fit = glmnet(X, y, alpha = 0, lambda = ridge.w$lambda.min)
betaS_ridge = data.frame(c(1:p), as.vector(ridge.w.fit$beta), 2*ridge_bs_sd)
colnames(betaS_ridge) = c("Feature", "Value", "Error")

en.w = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
en.w.fit = glmnet(X, y, alpha = 0.5, lambda = en.w$lambda.min)
betaS_en = data.frame(c(1:p), as.vector(en.w.fit$beta), 2*en_bs_sd)
colnames(betaS_en) = c("Feature", "Value", "Error")

rf.w.fit = randomForest(X, y, mtry = sqrt(p), importance = TRUE)
betaS_rf = data.frame(c(1:p), as.vector(rf.w.fit$importance[,1]), 2*rf_bs_sd)
colnames(betaS_rf) = c("Feature", "Value", "Error")

# Plot the estimated coefficients, and the importance of the parameters (Random Forest$importance)
# Before plotting, let's change the order of factors 
betaS_lasso$Feature = factor(betaS_lasso$Feature, levels = betaS_rf$Feature[order(betaS_rf$Value, decreasing = TRUE)])
betaS_ridge$Feature = factor(betaS_ridge$Feature, levels = betaS_rf$Feature[order(betaS_rf$Value, decreasing = TRUE)])
betaS_en$Feature = factor(betaS_en $Feature, levels = betaS_rf$Feature[order(betaS_rf$Value, decreasing = TRUE)])
betaS_rf$Feature = factor(betaS_rf$Feature, levels = betaS_rf$Feature[order(betaS_rf$Value, decreasing = TRUE)])

lasso.plot =  ggplot(betaS_lasso, aes(x=Feature, y=Value)) +
              geom_bar(stat = "identity", fill="#DBE2EF", colour="#52616b") +
              geom_errorbar(aes(ymin=Value-Error, ymax=Value+Error), width=.2) +
              labs(title = "Lasso") + theme_minimal()

ridge.plot =  ggplot(betaS_ridge, aes(x=Feature, y=Value)) +
              geom_bar(stat = "identity", fill="#DBE2EF", colour="#52616b") +
              geom_errorbar(aes(ymin=Value-Error, ymax=Value+Error), width=.2) +
              labs(title = "Ridge") + theme_minimal()

en.plot =  ggplot(betaS_en, aes(x=Feature, y=Value)) +
           geom_bar(stat = "identity", fill="#DBE2EF", colour="#52616b") +
           geom_errorbar(aes(ymin=Value-Error, ymax=Value+Error), width=.2) +
           labs(title = "Elastic Net") + theme_minimal()

rf.plot =  ggplot(betaS_rf, aes(x=Feature, y=Value)) +
           geom_bar(stat = "identity", fill="#DBE2EF", colour="#52616b") +
           geom_errorbar(aes(ymin=Value-Error, ymax=Value+Error), width=.2) +
           labs(title = "Random Forest") + theme_minimal()

grid.arrange(lasso.plot, ridge.plot, en.plot, rf.plot, nrow = 4)


####### Performance Summary #######
# R^2
R_squares = data.frame(rsq_train_lasso, rsq_test_lasso, rsq_train_ridge, rsq_test_ridge,
                       rsq_train_en, rsq_test_en, rsq_train_rf, rsq_test_rf)
colnames(R_squares) = c("Lasso Train","Lasso Test", "Ridge Train","Ridge Test",
                        "Elastic Net Train", "Elastic Net Test", "Random Forest Train", "Random Forest Test")
summary(R_squares)
# === A data-driven statistical model for predicting the critical temperature of a superconductor by Kam Hamidieh
# Accordting to Kam, test R^2 of multiple regression model based on whole data set is 0.74, test R^2 of XGBoost = 0.92

# Feature Importance

# Order by absolute value of coefficients in Lasso, Ridge and Elastic Net
# Let's see which are the top 10 "important" features under different models

betaS_lasso %>% arrange(desc(abs(Value)))
# Feature       Value     Error
# 1       15  28.3136771 23.758248
# 2       67 -26.5103404 15.922977
# 3       49  23.9576144 10.954577
# 4       17  23.6351300 17.799291
# 5       42  23.4186950 15.251746
# 6       44 -23.2769093 13.996768
# 7       76 -22.7846985 17.035985
# 8       25 -22.6411681 17.343831
# 9       47 -20.9284864 10.370829
# 10      69  19.6557748 19.11838

betaS_ridge %>% arrange(desc(abs(Value)))
# Feature        Value        Error
# 1       62  6.746847897 1.891981e+00
# 2       80 -5.959277562 1.605475e+00
# 3       70  5.424655814 2.651774e+00
# 4       44 -5.213927978 2.000876e+00
# 5       46 -5.076836994 0.000000e+00
# 6       64 -5.048979549 0.000000e+00
# 7        7  4.974750671 2.055383e+00
# 8        6  3.800937623 1.998685e+00
# 9       27  3.754066499 2.604745e+00
# 10       9  3.437459791 0.000000e+00

betaS_en %>% arrange(desc(abs(Value)))
# Feature         Value     Error
# 1       15  27.759393224 21.930370
# 2       67 -26.220696114 16.351790
# 3       76 -24.388631175 16.604806
# 4       42  23.837556070 15.323701
# 5       49  23.784755818 11.044373
# 6       44 -23.543612289 13.940614
# 7       17  22.908945517 17.276572
# 8       25 -22.860402165 15.865238
# 9       47 -20.860474066 10.441182
# 10      69  19.995697799 19.280979

betaS_rf %>% arrange(desc(abs(Value)))
# Feature      Value     Error
# 1       70 174.957692 57.101234
# 2       67 156.668011 50.252598
# 3       72 145.095403 43.285632
# 4       74 113.446086 60.968195
# 5       76 104.943394 43.009456
# 6       69  94.872605 37.508122
# 7       27  91.775481 35.384299
# 8       50  78.767170 28.088732
# 9       62  71.140355 22.549523
# 10      75  69.346644 29.631742


# RMSE (sqrt(mean((predicted - observed)^2)))

sqrt(mean(res_test_lasso^2)) # Lasso RMSE = 18.91527
sqrt(mean(res_test_ridge^2)) # Ridge RMSE = 19.59293
sqrt(mean(res_test_en^2)) # Elastic Net RMSE = 18.89045
sqrt(mean(res_test_rf^2)) # Random Forest RMSE = 13.76742

####### System Time #######

system.time({
lasso.time.cv = cv.glmnet(X_bs, y_bs, alpha = 1, nfolds = 10)
lasso.time.fit = glmnet(X_bs, y_bs, intercept = FALSE, alpha = 1, lambda = lasso.time.cv$lambda.min)
})
# user  system elapsed 
# 0.581   0.038   0.631 

# Ridge
system.time({
ridge.time.cv = cv.glmnet(X_bs, y_bs, alpha = 0, nfolds = 10)
ridge.time.fit = glmnet(X_bs, y_bs, intercept = FALSE, alpha = 1, lambda = ridge.time.cv$lambda.min)
})
# user  system elapsed 
# 0.167   0.014   0.183 

# Elastic Net (alpha = 0.5)
system.time({
en.time.cv = cv.glmnet(X_bs, y_bs, alpha = 0.5, nfolds = 10)
en.time.fit = glmnet(X_bs, y_bs, intercept = FALSE, alpha = 1, lambda = en.time.cv$lambda.min)
})
# user  system elapsed 
# 0.506   0.016   0.552 

# Random Forest
system.time({
rf.time = randomForest(X_bs, y_bs, mtry = sqrt(p), importance = TRUE) 
})
# user  system elapsed 
# 5.580   0.076   5.841 









