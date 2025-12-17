###113 ML
setwd("C:/Users/zzy/Desktop/Imagine/six/data/machine learning")
library(readxl)
TNBC <- read_xlsx("TNBC.xlsx")
expr_matrix <- as.matrix(TNBC[ , -1])
rownames(expr_matrix) <- TNBC[[1]]
set.seed(1234)
n_total  <- ncol(expr_matrix)
ratio    <- c(7, 1, 1, 1)         
frac     <- ratio / sum(ratio)      
idx_rand <- sample(n_total)        
idx_list <- split(idx_rand,
                  rep(1:4, times = round(n_total * frac))) |>
  (\(x) {
    x[[4]] <- c(x[[4]], setdiff(idx_rand, unlist(x)))
    x
  })()
sub_names <- c("Training", "Validation", "Test1", "Test2")
data_list <- lapply(idx_list, \(i) expr_matrix[ , i, drop = FALSE])
names(data_list) <- sub_names
out_dir <- file.path(getwd(), "raw_data")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
for (nm in sub_names) {
  write.csv(data_list[[nm]],
            file = file.path(out_dir, paste0(nm, ".csv")),
            row.names = TRUE)
}
cat("划分完成！文件已保存至：", out_dir, "\n")
work.path <- "C:/Users/zzy/Desktop/Imagine/six/data/machine learning"
setwd(work.path) 
code.path <- file.path(work.path, "Codes")
data.path <- file.path(work.path, "InputData")
res.path <- file.path(work.path, "Results")
fig.path <- file.path(work.path, "Figures")
if (!dir.exists(data.path)) dir.create(data.path)
if (!dir.exists(res.path)) dir.create(res.path)
if (!dir.exists(fig.path)) dir.create(fig.path)
if (!dir.exists(code.path)) dir.create(code.path)
library(openxlsx)
library(seqinr)
library(plyr)
library(randomForestSRC)
library(glmnet)
library(plsRglm)
library(gbm)
library(caret)
library(mboost)
library(e1071)
library(BART)
library(MASS)
library(snowfall)
library(xgboost)
library(ComplexHeatmap)
library(RColorBrewer)
library(pROC)
library(circlize)
source(file.path(code.path, "ML.R"))
FinalModel <- c("panML", "multiLogistic")[2]
Train_expr <- read.table(file.path(data.path, "Training_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
Train_class <- read.table(file.path(data.path, "Training_class.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
comsam <- intersect(rownames(Train_class), colnames(Train_expr))
Train_expr <- Train_expr[,comsam]; Train_class <- Train_class[comsam,,drop = F]
Test_expr <- read.table(file.path(data.path, "Testing_expr.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
Test_class <- read.table(file.path(data.path, "Testing_class.txt"), header = T, sep = "\t", row.names = 1,check.names = F,stringsAsFactors = F)
comsam <- intersect(rownames(Test_class), colnames(Test_expr))
Test_expr <- Test_expr[,comsam]; Test_class <- Test_class[comsam,,drop = F]
comgene <- intersect(rownames(Train_expr),rownames(Test_expr))
Train_expr <- t(Train_expr[comgene,])
Test_expr <- t(Test_expr[comgene,])
Train_set = scaleData(data = Train_expr, centerFlags = T, scaleFlags = T) 
names(x = split(as.data.frame(Test_expr), f = Test_class$Cohort))
Test_set = scaleData(data = Test_expr, cohort = Test_class$Cohort, centerFlags = T, scaleFlags = T)
summary(apply(Train_set, 2, var))
summary(apply(Test_set, 2, var))
lapply(split(as.data.frame(Test_set), Test_class$Cohort), function(x) summary(apply(x, 2, var)))
methods <- read.xlsx(file.path(code.path, "methods.xlsx"), startRow = 2)
methods <- methods$Model
methods <- gsub("-| ", "", methods)
classVar = "outcome"
min.selected.var = 5
Variable = colnames(Train_set)
preTrain.method =  strsplit(methods, "\\+") 
preTrain.method = lapply(preTrain.method, function(x) rev(x)[-1]) 
preTrain.method = unique(unlist(preTrain.method))
preTrain.var <- list() 
set.seed(seed = 123) 
for (method in preTrain.method){
  preTrain.var[[method]] = RunML(method = method, 
                                 Train_set = Train_set,
                                 Train_label = Train_class, 
                                 mode = "Variable",      
                                 classVar = classVar) 
}
preTrain.var[["simple"]] <- colnames(Train_set)
model <- list() 
set.seed(seed = 123) 
Train_set_bk = Train_set 
for (method in methods){
  cat(match(method, methods), ":", method, "\n")
  method_name = method 
  method <- strsplit(method, "\\+")[[1]] 
  if (length(method) == 1) method <- c("simple", method) 
  Variable = preTrain.var[[method[1]]] 
  Train_set = Train_set_bk[, Variable]   
  Train_label = Train_class            
  model[[method_name]] <- RunML(method = method[2],        
                                Train_set = Train_set,     
                                Train_label = Train_label, 
                                mode = "Model",            
                                classVar = classVar)       
  if(length(ExtractVar(model[[method_name]])) <= min.selected.var) {
    model[[method_name]] <- NULL
  }
}
Train_set = Train_set_bk; rm(Train_set_bk) 
saveRDS(model, file.path(res.path, "model.rds")) 
if (FinalModel == "multiLogistic"){
  logisticmodel <- lapply(model, function(fit){
    tmp <- glm(formula = Train_class[[classVar]] ~ .,
               family = "binomial", 
               data = as.data.frame(Train_set[, ExtractVar(fit)]))
    tmp$subFeature <- ExtractVar(fit) 
    return(tmp)
  })
}
saveRDS(logisticmodel, file.path(res.path, "logisticmodel.rds"))
model <- readRDS(file.path(res.path, "model.rds"))
methodsValid <- names(model)
RS_list <- list()
for (method in methodsValid){
  RS_list[[method]] <- CalPredictScore(fit = model[[method]], 
                                       new_data = rbind.data.frame(Train_set,Test_set)) 
}
RS_mat <- as.data.frame(t(do.call(rbind, RS_list)))
write.table(RS_mat, file.path(res.path, "RS_mat.txt"),sep = "\t", row.names = T, col.names = NA, quote = F)
Class_list <- list()
for (method in methodsValid){
  Class_list[[method]] <- PredictClass(fit = model[[method]], 
                                       new_data = rbind.data.frame(Train_set,Test_set))
}
Class_mat <- as.data.frame(t(do.call(rbind, Class_list)))
write.table(Class_mat, file.path(res.path, "Class_mat.txt"),
            sep = "\t", row.names = T, col.names = NA, quote = F)
fea_list <- list()
for (method in methodsValid) {
  fea_list[[method]] <- ExtractVar(model[[method]])
}
fea_df <- lapply(model, function(fit){
  data.frame(ExtractVar(fit))
})
fea_df <- do.call(rbind, fea_df)
fea_df$algorithm <- gsub("(.+)\\.(.+$)", "\\1", rownames(fea_df))
colnames(fea_df)[1] <- "features"
write.table(fea_df, file.path(res.path, "fea_df.txt"),
            sep = "\t", row.names = F, col.names = T, quote = F)
AUC_list <- list()
for (method in methodsValid){
  AUC_list[[method]] <- RunEval(fit = model[[method]],     
                                Test_set = Test_set,
                                Test_label = Test_class,   
                                Train_set = Train_set,
                                Train_label = Train_class,
                                Train_name = "Training Set",
                                cohortVar = "Cohort",
                                classVar = classVar)
}
AUC_mat <- do.call(rbind, AUC_list)
write.table(AUC_mat, file.path(res.path, "AUC_mat.txt"),
            sep = "\t", row.names = T, col.names = T, quote = F)
AUC_mat <- read.table(file.path(res.path, "AUC_mat.txt"),sep = "\t", row.names = 1, header = T,check.names = F,stringsAsFactors = F)
avg_AUC <- apply(AUC_mat, 1, mean)
avg_AUC <- sort(avg_AUC, decreasing = T)
AUC_mat <- AUC_mat[names(avg_AUC), ]
fea_sel <- fea_list[[rownames(AUC_mat)[1]]]
avg_AUC <- as.numeric(format(avg_AUC, digits = 3, nsmall = 3))
if(ncol(AUC_mat) < 3) { 
  CohortCol <- c("red","blue") 
} else { 
  CohortCol <- brewer.pal(n = ncol(AUC_mat), name = "Paired")
}
names(CohortCol) <- colnames(AUC_mat)
cellwidth = 2
cellheight = 0.5
hm <- SimpleHeatmap(AUC_mat, 
                    avg_AUC,
                    CohortCol, "steelblue", 
                    cellwidth = cellwidth, cellheight = cellheight, 
                    cluster_columns = F, cluster_rows = F) 

png(file.path(fig.path, "113 machine learning.png"),
    width  = (cellwidth * ncol(AUC_mat) + 2) * 300,
    height = (cellheight * nrow(AUC_mat) * 0.42) * 300,
    res    = 300)
draw(hm)
library(xgboost)
library(dplyr)
library(ggplot2)
model <- readRDS(file.path(res.path, "model.rds"))
method_name <- "Lasso+XGBoost"
if (!method_name %in% names(model)) {
  stop("模型中未找到 Lasso+XGBoost，请检查 methods.xlsx 或 model 列表")
}
fit <- model[[method_name]]
selected_vars <- ExtractVar(fit)
Train_set_selected <- Train_set[, selected_vars, drop = FALSE]
Train_label_binary <- as.numeric(Train_class[[classVar]])   
dtrain <- xgb.DMatrix(data = as.matrix(Train_set_selected), label = Train_label_binary)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)
importance <- xgb.importance(model = xgb_model)
write.csv(importance, file.path(res.path, "Lasso_XGBoost_importance.csv"), row.names = FALSE)
setwd('C:/Users/zzy/Desktop/Imagine/six/data')
library(ggplot2)
mtcars <- read.csv('mtcars.csv', header = TRUE)
df<-data.frame(mtcars)
colnames(df)[1] <- "Names"
rownames(df) <- NULL
colnames(df) <- make.unique(colnames(df))
colors1=c("#E64B35FF","#4DBBD5FF","#00A087FF","#3C5488FF","#F39B7FFF","#8491B4FF","#91D1C2FF","#DC0000FF","#7E6148FF","#54D9DE")
p <- ggplot(df, aes(x = Names, y = mpg))+
  geom_segment(aes(x = Names, xend = Names, y = 0, yend = mpg), color = "gray")+
  geom_point(aes(x = Names, y = mpg, color = Names), size = 5)+
  geom_point(aes(x = Names, y = mpg, color = Names), shape = 1, size = 6)+
  geom_text(aes(label = mpg, x = Names, y = mpg + 0.7), hjust = -0.2, size = 3)+
  scale_color_manual(values = colors1)+
  scale_y_continuous(expand = c(0, 0), limits = c(0, 0.3))+
  coord_flip()+
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(size = 0.7),
        axis.ticks = element_line(size = 0.7),
        axis.ticks.length = unit(0.1, "cm"),
        axis.text.y = element_text(size = 12, colour = "black"),
        axis.text.x = element_text(size = 11, colour = "black"),
        axis.title.x = element_text(size = 12, colour = "black"),
        axis.title.y = element_blank(),
        legend.position = "none")+
  labs(y = "Importance")
ggsave("lollipop.png", p, width = 7, height = 5, dpi = 300)