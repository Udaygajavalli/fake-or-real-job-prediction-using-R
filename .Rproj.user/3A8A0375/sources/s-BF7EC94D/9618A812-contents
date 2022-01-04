library(plumber)
library(mlr)
rf_model <- readRDS("rf_model.rds")
library(shiny)
library(shinythemes)
library(caret)
library(lubridate)
library(data.table)
library(tidytext)
library(tm)
library(SnowballC)
library(wordcloud)
library(stopwords)
library(e1071)

#* @filter cors
cors <- function(res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods"," GET,HEAD,PUT,PATCH,POST,DELETE")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Origin, xxx")
  plumber::forward()
}

#* @param jd
#* @get /predict
predictions <- function(jd){
  str <- as.character(jd)
  corpus <- Corpus(VectorSource(jd))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords())
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  freq <- DocumentTermMatrix(corpus)
  sparse <- removeSparseTerms(freq, 0.995)
  sparse_df <- as.data.frame(as.matrix(sparse))
  colnames(sparse_df) <- make.names(colnames(sparse_df))
  col_names = colnames(test_data)
  sparse_df[col_names[!(col_names %in% colnames(sparse_df))]] = 0
  pred <- predict(rf_model, newdata = sparse_df)
  # res = list(output=pred)
  return(pred)
}