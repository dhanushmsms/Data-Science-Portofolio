#data explorartion 

library(tidyverse)
library(ggplot2)

getwd()
setwd('/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Marketing Analytics/Ass_1')

data <- read_csv('/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Marketing Analytics/Ass_1/Assignment1-data (1).csv')

names(data)

attach(data)

summary(InvoiceNo)

library(ggplot2)

#Quantity 
library(ggplot2)
ggplot(data, aes(x = Quantity)) +
  geom_boxplot(outlier.color = "red") +
  labs(title = "Box Plot of Quantity") +
  theme_minimal()

#Unit Price
ggplot(data, aes(x = `UnitPrice`)) +
  geom_boxplot(outlier.color = "red") +
  labs(title = "Box Plot of Unit Price") +
  theme_minimal()

#Return Rate
ggplot(data, aes(x = `ReturnRate`)) +
  geom_boxplot(outlier.color = "red") +
  labs(title = "Box Plot of Return Rate") +
  theme_minimal()

#Age
ggplot(data, aes(x = Age)) +
  geom_boxplot(outlier.color = "red") +
  labs(title = "Box Plot of Age") +
  theme_minimal()

#Income
ggplot(data, aes(x = Income)) +
  geom_boxplot(outlier.color = "red") +
  labs(title = "Box Plot of Income") +
  theme_minimal()

#marriage 
ggplot(data, aes(x = Married, fill = factor(Married))) +
  geom_bar(width = 0.75, fill = c("red", "black")) +
  labs(title = "Marital Status", x = "Married", y = "Count") +
  theme_minimal()


# Convert 'Work' variable to factor
n_data<- data
n_data$Work <- factor(n_data$Work)

ggplot(n_data, aes(x = Work, fill = Work)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Distribution of Work Categories", x = "Work", y = "Count") +
  theme_minimal()



#Education
ggplot(data, aes(x = Edcation, fill = factor(Edcation))) +
  geom_bar(fill = c("blue","lightgreen","pink")) +
  labs(title = "Distribution of Education Levels", x = "Education", y = "Count") +
  theme_minimal()


install.packages("factoextra")
install.packages("NbClust")
install.packages("fpc")
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE)
library(lubridate)
library(factoextra)
library(NbClust)

library(fpc)
library(cluster)
library(SmartEDA)
library(psych)
library(ggplot2)
library(dplyr, warn.conflicts = FALSE, quietly = TRUE)


# Reading the data
data <- read_csv(file.choose())
View(data)


#######################################################################################
get_mode <- function(x) {
  tab <- table(x)
  mode <- names(tab)[which.max(tab)]
  return(mode)
}

# Imputating mean for the numeric data and mode for the categorical data.
data <- data %>% 
  group_by(CustomerID) %>% 
  mutate(Age = round(mean(Age)),
         Income = round(mean(Income)),
         Married = get_mode(Married),
         Work = get_mode(Work),
         Edcation = get_mode(Edcation),
         ZipCode = get_mode(ZipCode)) %>%
  ungroup()

# A) Data Preparation and understanding.
# A.1) Omitted the CustomerID.

Missing_values_Columns <- colSums(is.na(data))
barplot(Missing_values_Columns)
data <- na.omit(data)

# A.2) Return Rate with greater than 1 will be omitted.
data <- data %>% filter(ReturnRate < 1)

# A.3) change the data type of the characters.
data$Work <- as.factor(data$Work)
data$Married <- as.factor(data$Married)
data$Edcation <- as.factor(data$Edcation)
data$ZipCode <- as.factor(data$ZipCode)

# A.3.1) Created the variables year and month to include it in the base variables. 
data$Year <- year(data$InvoiceDate)
data$Month <- month(data$InvoiceDate)

# B) Descriptive Statistics.
str(data)


########################################################################################
# C) Cluster Analysis.
# C.1) Finding the Optimal Number of cluster using Base Descriptor.

base_variables <- data %>% dplyr::select(Quantity, ReturnRate, UnitPrice)
base_var_scale <- scale(base_variables)

### Hclust
hir_complete <- hclust(dist(base_var_scale, method = "euclidean"), method = "complete")
hir_centroid <- hclust(dist(base_var_scale, method = "euclidean"), method = "centroid")
hir_avg <- hclust(dist(base_var_scale, method = "euclidean"), method = "average")
hir_single <- hclust(dist(base_var_scale, method = "euclidean"), method = "single")


y_complete <- sort(hir_complete$height, decreasing = TRUE)[1:10]
y_centroid <- sort(hir_centroid$height, decreasing = TRUE)[1:10]
y_avg <- sort(hir_avg$height, decreasing = TRUE)[1:10]
y_single <- sort(hir_single$height, decreasing = TRUE)[1:10]
x <- c(1:10)


# Create the data frame
df <- data.frame(x = x,
                 y_complete = y_complete,
                 y_centroid = y_centroid,
                 y_avg = y_avg,
                 y_single = y_single)


# Visualising the results
# Define aesthetically pleasing colors
my_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00")

# Plot with updated colors
ggplot(df) +
  geom_line(aes(x = x, y = y_complete, color = "Complete")) +
  geom_line(aes(x = x, y = y_centroid, color = "Centroid")) +
  geom_line(aes(x = x, y = y_avg, color = "Average")) +
  geom_line(aes(x = x, y = y_single, color = "Single")) +
  geom_point(aes(x = x, y = y_complete), color = "#1f78b4") +
  geom_point(aes(x = x, y = y_centroid), color = "#33a02c") +
  geom_point(aes(x = x, y = y_avg), color = "#e31a1c") +
  geom_point(aes(x = x, y = y_single), color = "#ff7f00") +
  scale_color_manual(values = my_colors) +
  labs(color = "Linkage Method", title = "Optimal No. of Clusters", 
       x = "No. of Clusters", y = "Distance") +
  scale_x_continuous(breaks = 1:10)+
  theme_minimal()



plot(hir_complete)


# D) After the thorough consideration we have come to a conclusion that the optimal number of clusters for our data set is 6.
# Hence the optimal no. of cluster is 3.

# D.1) KMeans algorithm.
set.seed(40412492)
km.res <- kmeans(x = base_var_scale, centers = 3, iter.max = 500, nstart = 1000)

table(km.res$cluster)
km.res_cluster <- fviz_cluster(km.res, 
                               base_var_scale, 
                               ellipse = TRUE,
                               star.plot =TRUE,
                               ggtheme = theme_minimal(), 
                               palette = "jco", 
                               ellipse.type = "norm") +
  labs(title = "KMeans Clustering")

km.res_cluster





########################################################################################
# cluster is added as the column in the data and the new data is named as data_clustered.
data_clustered <- cbind(data, cluster = km.res$cluster)

# Assuming 'data_clustered' is your dataframe
write.csv(data_clustered, file = file.choose(new = TRUE), row.names = FALSE)

# E)  Discriminant Analysis
library(MASS)

# To check which discriminant functions are significant.
# We are running the classification algorithm LDA to predict the classes using the Descriptor variables.
fit <- lda(cluster ~ Married + Age + Income + Work + Edcation, data = data_clustered)
fit

ldaPred <- predict(fit, data_clustered)
ld <- ldaPred$x

fit.predict <- ldaPred$class

#confusion Matrix
confusionMatrix(fit.predict, as.factor(data_clustered$cluster))

#ANOVA
anova(lm(ld[,1] ~ data_clustered$cluster ))
anova(lm(ld[,2] ~ data_clustered$cluster ))

summary_revenue <- data_clustered %>% group_by(cluster) %>% 
  summarise(total_Customers = n(),total_profits = sum(Quantity*UnitPrice)) %>% 
  mutate("Revenue/Segment" = total_profits/total_Customers)

summary_revenue

## Check Disciminant Model Fit
pred.seg <- predict(fit)$class
tseg <- table(data_clustered$cluster, pred.seg)
tseg # print table
sum(diag(tseg))/nrow(data_clustered) # print percent correct


#############################################################################################################
# RFM

library(rfm)
analysis_date <- lubridate::as_date("2022-01-31")

rfm <- data %>% group_by(CustomerID) %>% 
  summarise(total_trans = n(), 
            total_revenue = sum(UnitPrice*Quantity), 
            recency_days = min(analysis_date - date(InvoiceDate)))

rfm$recency_days <- as.numeric(rfm$recency_days)
colnames(rfm)[1] <- "customer_id"


rfm_table <- rfm_table_customer(data = rfm,
                                customer_id = customer_id,
                                n_transactions = total_trans,
                                recency = recency_days,
                                total_revenue = total_revenue,
                                analysis_date = analysis_date,
                                recency_bins = 5,
                                frequency_bins = 5,
                                monetary_bins = 5)


table(rfm_table$rfm$recency_score)
table(rfm_table$rfm$frequency_score)
table(rfm_table$rfm$monetary_score)


# Write the RFM table to a CSV file
write.csv(rfm_table$rfm, "rfm_analysis_results.csv", row.names = FALSE)

plot(rfm)

rfm_data <- read_csv('rfm_analysis_results.csv')

library(ggplot2)

# Recency Score Distribution
ggplot(rfm_table$rfm, aes(x = recency_score)) +
  geom_bar(fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Recency Scores", x = "Recency Score", y = "Count")

# Frequency Score Distribution
ggplot(rfm_table$rfm, aes(x = frequency_score)) +
  geom_bar(fill = "green", alpha = 0.7) +
  labs(title = "Distribution of Frequency Scores", x = "Frequency Score", y = "Count")

# Monetary Score Distribution
ggplot(rfm_table$rfm, aes(x = monetary_score)) +
  geom_bar(fill = "red", alpha = 0.7) +
  labs(title = "Distribution of Monetary Scores", x = "Monetary Score", y = "Count")
#######################
# Scatter plot of Frequency vs Monetary
ggplot(rfm_table$rfm, aes(x = frequency_score, y = monetary_score)) +
  geom_point(aes(color = recency_score), alpha = 0.6) +
  scale_color_gradient(low = "yellow", high = "red") +
  labs(title = "Frequency vs. Monetary by Recency Score",
       x = "Frequency Score",
       y = "Monetary Score",
       color = "Recency Score")
