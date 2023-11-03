# Tharawit Disyawongs
# MovieLens Project
# HarvardX PH125.9x Data Science: Capstone

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
#Split edx data into test and train sets
##########################################################
set.seed(1)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
tmp <- edx[test_index,]
train_set <- edx[-test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- tmp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(tmp, test_set)
train_set <- rbind(train_set, removed)

# Remove unused objects
rm(tmp,removed,test_index)

##########################################################
# Model 1: Mean of all ratings. Y = mu+e
##########################################################
mu <- mean(train_set$rating)
mu

model_1_rmse <- RMSE(test_set$rating, mu)
results <- data_frame(Method = "Model 1: Mean of all ratings", RMSE = round(model_1_rmse,7))
results %>% knitr::kable()

##########################################################
#Model 2: Mean + movie effect. Y= mu+bi+e
##########################################################

#Observe the distribution of ratings grouped by movies. 
edx %>% group_by(movieId) %>% summarize(mean(rating)) %>% pull() %>% hist(main=NULL)

#Create movie effect (bi), add to the model and calculate RMSE
bi <- train_set %>% group_by(movieId) %>% summarize(bi=mean(rating-mu))
bi %>% ggplot(aes(x=bi)) + geom_histogram(binwidth=0.1,color="black", fill="white")

model_2_rmse <- test_set %>% left_join(bi, by = "movieId") %>% mutate(pred=mu+bi) %>% summarize(rmse = RMSE(rating, pred)) %>% pull(rmse)
results <- rbind(results,c("Model 2: Mean + movie effect", round(model_2_rmse,7)))
results %>% knitr::kable()

##########################################################
#Model 3: Mean + movie effect + user effect. Y= mu+bi+bu+e
##########################################################

#Observe the distribution of average ratings grouped by users. 
edx %>% group_by(userId) %>% summarize(mean(rating)) %>% pull() %>% hist(main=NULL)

#Create user effect (bu), add to the model and calculate RMSE
bu <- train_set %>% left_join(bi,by="movieId") %>% group_by(userId) %>% summarize(bu=mean(rating-mu-bi))
bu %>% ggplot( aes(x=bu)) + geom_histogram(binwidth=0.1,color="black", fill="white")

model_3_rmse <- test_set %>% left_join(bi, by = "movieId") %>% left_join(bu, by = "userId")  %>% mutate(pred=mu+bi+bu) %>% summarize(rmse = RMSE(rating, pred)) %>% pull(rmse)
results <- rbind(results,c("Model 3: Mean + movie effect + user effect", round(model_3_rmse,7)))
results %>% knitr::kable()

##########################################################
#Model 4: Regularization on movie and user effects
##########################################################

#Movies with high ratings but actually got rated by very few users
edx %>% group_by(movieId) %>% summarize(avg_rating=mean(rating),n=n()) %>% filter(avg_rating>4.5 ) %>% left_join(edx, by = "movieId") %>% select(movieId,title, avg_rating, n) %>% distinct() %>% arrange(desc(avg_rating))

#Choosing the tuning values (lambdas) for regularization, and calculate RMSE
lambdas <- seq(0, 6, 0.25)
rmses <- sapply(lambdas,function(x){
  bi <- train_set %>% group_by(movieId) %>% summarize(bi=sum(rating-mu)/(n()+x))
  bu <- train_set %>% left_join(bi,by="movieId") %>% group_by(userId)  %>% summarize(bu=sum(rating-mu-bi)/(n()+x))
  pred_ratings <- test_set %>% left_join(bi, by = "movieId")%>% left_join(bu, by = "userId")  %>% mutate(pred=mu+bi+bu) %>% pull(pred)
  return(RMSE(test_set$rating, pred_ratings))
})
qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

model_4_rmse <- min(rmses)
results <- rbind(results,c("Model 4: Regularization on movie and user effects", round(model_4_rmse,7)))
results %>% knitr::kable()

##########################################################
#Model 5: Matrix factorization
##########################################################
library(recosystem)
set.seed(1)
train_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
reco <- Reco()
reco$train(train_reco) 

results_reco <- reco$predict(test_reco, out_memory())
model_5_rmse <- RMSE(results_reco, test_set$rating)
results <- rbind(results,c("Model 5: Matrix factorization", round(model_5_rmse,7)))
results %>% knitr::kable()

##########################################################
#Model 6: Matrix factorization with tuning parameters
##########################################################
set.seed(1)
opts_tune <- reco$tune(train_reco, opts = list(costp_l2 = c(0.01, 0.1), # user regularization
                                            costq_l2 = c(0.01, 0.1), # movie regularization
                                            nthread = 1)) # number of thread

reco$train(train_reco, opts = opts_tune$min) 
results_reco <- reco$predict(test_reco, out_memory())

model_6_rmse <- RMSE(results_reco, test_set$rating)
results <- rbind(results,c("Model 6: Matrix factorization with tuning parameters", round(model_6_rmse,7)))
results %>% knitr::kable()

##########################################################
#Final validation: Run the final model (model 6) on final_holdout_test
##########################################################
final_holdout_reco <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId, rating = rating))
pred_reco <- reco$predict(final_holdout_reco, out_memory())
final_rmse <- RMSE(final_holdout_test$rating, pred_reco)
results <- rbind(results,c("Final validation: model 6 on final_holdout_test", round(final_rmse,7)))
results %>% knitr::kable()
