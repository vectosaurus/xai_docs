# read in all packages
library(data.table)
library(caret)
library(dummy)
library(nnet)
library(randomForest)
library(RWeka)
# set options
options(warn = -1)
# read data
data <- as.data.frame(fread("train.arff.csv"))

# check data types
dtypes <- as.matrix(sapply(data, class))
num_levels <- as.matrix(sapply(data, function(x) {
    length(unique(x))
}))
data_class_changed <- as.data.frame(sapply(data, as.factor))
target_levels <- as.matrix(sapply(data_class_changed, class))

num_nas <- apply(data_class_changed, 2, function(x) sum(x == " "))
colnames(data_class_changed)[7] <- "category_type"
target_levels_dist <- 100 * table(data_class_changed$class)/dim(data_class_changed)[1]


target_variable = "category_type"
balanced_col_suffx <- "_blncd"

lvl_counts <- as.data.frame(table(data_class_changed[,target_variable]))
count_max <- max(lvl_counts$Freq)
curr_lvl <- "vgood"
curr_lvl_count <- sum(data_class_changed[,target_variable] == curr_lvl)
curr_prec_over <- 100*as.integer(count_max/curr_lvl_count)

target_variable = "category_type"
balanced_col_suffx <- "_blncd"
adj_target_variable <- paste0(target_variable,balanced_col_suffx)
# adj_target_variable
# sapply(data_class_changed,class)
library(DMwR)
data_class_changed[adj_target_variable] <- as.character(data_class_changed[,target_variable])
data_class_changed[data_class_changed[adj_target_variable] == curr_lvl,adj_target_variable] = 1
data_class_changed[data_class_changed[adj_target_variable] != "1",adj_target_variable] = 0
data_class_changed[,adj_target_variable] <- as.factor(data_class_changed[,adj_target_variable])
# table(data_class_changed[,adj_target_variable])
# table(data_class_changed[,target_variable])

model_cols <- names(data_class_changed)[!(names(data_class_changed) == target_variable)]
# model_cols
# names(data_class_changed[,model_cols])
tail(data_class_changed)
dmSmote<-SMOTE(target_variable ~ . , data_class_changed,k=5,perc.over = curr_prec_over, perc.under = 100)
# table(dmSmote[,target_variable])
# table(dmSmote[,adj_target_variable])