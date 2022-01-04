library(ggplot2)
library(dplyr)
library(tidyverse)
library(ggthemes)
library(skimr)
library(DataExplorer)
library(ggpubr)
model <- readRDS("rf_model.rds")


data <- read_csv("fake_job_postings.csv")
plot_str(data, type = "diagonal")
skim(data)
plot_intro(data, title = "Data Quality" )
plot_missing(data, title = "Missing data")
data <- data %>% select(-salary_range)
data %>% group_by(title) %>% summarize(Freq = n()) %>% arrange(desc(Freq))

data_location_clean <- gsub("(.*),.*", "\\1", data$location)
data_location_clean_countries <- gsub("(.*),.*", "\\1", data_location_clean)
data_location_clean_countries <- data.frame(Country = data_location_clean_countries)


data_location_clean_countries %>% group_by(Country) %>% summarize(Freq= n()) %>% arrange(desc(Freq)) %>% slice(1:5,7:11) %>%
  ggplot(aes(x= reorder(Country, -Freq), y = Freq)) + geom_bar(stat = "identity", color = "black", fill = "steelblue") + theme_bw() +
  labs(title = "Top 10 countries with number of jobs listed",
       x = "Job Country",
       y = "Count") +
  geom_text(aes(label=round(Freq,0)), vjust=-0.5) + theme(axis.text.x=element_text(size=10, angle=0,hjust=0.5,vjust=1))
data$country <- data_location_clean_countries

data$telecommuting <- if_else(data$telecommuting == 1, "Required", "Not Required")
data %>% group_by(telecommuting) %>% summarize(Freq = n()) %>% arrange(desc(Freq)) %>% 
  ggplot(aes(x = reorder(telecommuting, -Freq), y = Freq)) + geom_bar(stat = "identity", fill = "steelblue", color = "black") + 
  theme_bw() + labs(title = "Jobs that require telecommuting",
                    x = "Telecommuting",
                    y = "Count") +
  geom_text(aes(label=round(Freq,0)), vjust= - 0.5) + theme(axis.text.x=element_text(size=10, angle=90,hjust=0.5,vjust=1))


data %>% group_by(employment_type) %>% summarize(Freq = n()) %>% arrange(desc(Freq)) %>% slice(1,3:6) %>%
  ggplot(aes(x = reorder(employment_type, -Freq), y = Freq, fill = employment_type)) + geom_bar(stat = "identity",  color = "black") + 
  theme_bw() + labs(title = "Employment types",
                    x = "Type",
                    y = "Count",
                    fill = "Employment Type") +
  geom_text(aes(label=round(Freq,0)), vjust= - 0.5) + theme(axis.text.x=element_text(size=10, angle=90,hjust=0.5,vjust=1))

fraud_job_data <- data %>% filter(fraudulent == 1)
fraud_job_data %>% group_by(department) %>% summarize(Freq = n()) %>% arrange(desc(Freq)) %>% slice(2:11) %>% 
  ggplot(aes(x = reorder(department, -Freq), y = Freq)) + geom_bar(stat = "identity", color = "black", fill = "purple") + 
  theme_bw() + labs(title = "Top 10 departments for a fraud job",
                    x = "Department",
                    y = "Count") +
  geom_text(aes(label=round(Freq,0)), vjust= -0.2) + theme(axis.text.x=element_text(size=10, angle=90,hjust=0.5,vjust=1))

##install.packages(c('caret','lubridate','data.table','tidytext','tm','Snowballc','wordcloud','stopwords','e1071'))
library(caret)
library(lubridate)
library(data.table)
library(tidytext)
library(tm)
library(SnowballC)
library(wordcloud)
library(stopwords)
library(e1071)
data_corpus <- Corpus(VectorSource(data$description))
data_corpus <- tm_map(data_corpus, removePunctuation)
data_corpus <- tm_map(data_corpus, removeWords, stopwords(kind = "en"))
data_corpus <- tm_map(data_corpus, stripWhitespace)
data_corpus <- tm_map(data_corpus, stemDocument)
frequencies <- DocumentTermMatrix(data_corpus)
sparse_data <- removeSparseTerms(frequencies, 0.995)
sparse_data_df <- as.data.frame(as.matrix(sparse_data))
colnames(sparse_data_df) <- make.names(colnames(sparse_data_df))
sparse_data_df$fraudulent <- data$fraudulent
colnames(sparse_data_df) <- make.unique(colnames(sparse_data_df), sep = "_")
set.seed(123)
split_data <- createDataPartition(y = sparse_data_df$fraudulent, times = 1, p = 0.85, list= FALSE)
train_data <- sparse_data_df[split_data, ]
test_data <- sparse_data_df[-split_data, ]
train_data$fraudulent = as.factor(train_data$fraudulent)
test_data$fraudulent = as.factor(test_data$fraudulent)
set.seed(1111)
trcontrol<- trainControl(method = "repeatedcv", number=2, repeats=1  , search="random", verboseIter = TRUE)
grid <-data.frame(mtry = c(100))
set.seed(1122)
rf_model <- train(fraudulent ~ ., method = "rf", data = train_data, ntree = 200, trControl = trcontrol,tuneGrid = grid)
rf_prediction <- predict(rf_model, newdata = test_data)
confMatrix_rf <- confusionMatrix(rf_prediction, test_data$fraudulent)
confMatrix_rf

saveRDS(rf_model, "rf_model.rds")


predict_custom <- function(str){
  corpus <- Corpus(VectorSource(str))
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
  print(pred)
  
}




