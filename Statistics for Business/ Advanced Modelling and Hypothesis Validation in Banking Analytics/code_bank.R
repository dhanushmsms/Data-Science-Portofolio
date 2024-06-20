#load libraries neccessary
library(tidyverse)
library(readxl)
library(dplyr)
#check the current working directory
getwd()
#set the working directory
setwd("/Users/dhanush/Desktop/Bussiness analytics /Stas/Ass 2")
#import the given
#attaching the provided data named "term.xlxs" using "read_excel" function and attaching it
to variable named data
data <- read_excel("term.xlsx")
#decprtive analysis
summary(data) # provided the summary of the whole data
unique(data) # gives us the uniques value in the data
names(data) # provides us the name of all the variables given
#1)descpritve analysis of ID
summary(data$ID) # summarises ID
sum(is.na(data$ID))#finds the total sum of null values in ID
#2)descrptive analysis of age
summary(data$age)#gives mean mode median of the data
sum(is.na(data$age))
# Create a histogram for 'age'
ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "lightgray", color = "black") +
  labs(title = "Distribution of Age",
       x = "Age",
       y = "Frequency") +
  theme_minimal()
#3)descpritive analysis of occupation
summary(data$occupation)#tells it is catergorical variables
unique(data$occupation)#gives the unique entries
count(data,occupation)#cout of each occurrence
32
# Load necessary libraries
library(ggplot2)
# Create a bar plot for 'occupation'
ggplot(data = data, aes(x = occupation)) +
  geom_bar(fill = "lightgray", color = "black") +
  labs(title = "Distribution of Occupation",
       x = "Occupation",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#4)descpritive analysis of marital status
summary(data$marital_status)#tells it is catergorical variables
unique(data$marital_status)#gives the unique entries
count(data,marital_status)#cout of each occurrence and there is error NA
# Create a bar plot for 'marital_status'
ggplot(data = data, aes(x = marital_status)) +
  geom_bar(fill = "#008080", color = "black") +
  labs(title = "Distribution of Martial status ",
       x = "marital_status",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#5)descpritive analysis of education_level
summary(data$education_level)#tells it is catergorical variables
unique(data$education_level)#gives the unique entries
count(data,education_level)#cout of each occurrence
# Create a bar plot for 'education_level'
ggplot(data = data, aes(x = education_level)) +
  geom_bar(fill = "lightgray", color = "black") +
  labs(title = "Distribution of education_level",
       x = "education_level",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#6)descpritive analysis of credit_default
summary(data$credit_default)#tells it is catergorical variables
unique(data$credit_default)#gives the unique entries
count(data,credit_default)#cout of each occurrence
# Create a bar plot for 'credit_default'
33
ggplot(data = data, aes(x = credit_default)) +
  geom_bar(fill = "#008080", color = "black") +
  labs(title = "Distribution of credit_default status ",
       x = "credit_default",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#7)descpritive analysis of housing_loan
summary(data$housing_loan)#tells it is catergorical variables
unique(data$housing_loan)#gives the unique entries
count(data,housing_loan)#cout of each occurrence
#pie chart for "housing_loan"
ggplot(data = data, aes(x = "", fill = housing_loan)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("#D3D3D3", "#008080", "#FFA500")) + # Light grey, teal, and
  another color of choice
labs(title = "Housing Loan Distribution",
     fill = "Housing Loan",
     x = NULL, y = NULL) +
  theme_void() +
  theme(legend.position = "bottom")
#8)descpritive analysis of personal_loan
summary(data$personal_loan)#tells it is catergorical variables
unique(data$personal_loan)#gives the unique entries
count(data,personal_loan)#cout of each occurrence
#pie chart for "personal loan"
ggplot(data = data, aes(x = "", fill = personal_loan)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("#D3D3D3", "#008080", "#FFA500")) + # Light grey, teal, and
  another color of choice
labs(title = "Personal Loan Distribution",
     fill = "Personal Loan",
     x = NULL, y = NULL) +
  theme_void() +
  theme(legend.position = "bottom")
#9)descpritive analysis of contact_method
summary(data$contact_method)#tells it is catergorical variables
unique(data$contact_method)#gives the unique entries
count(data,contact_method)#cout of each occurrence
34
# Create a bar plot for 'contact_method'
ggplot(data = data, aes(x = contact_method)) +
  geom_bar(fill = "lightgray", color = "black") +
  labs(title = "Distribution of Contact Method",
       x = "Contact method",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#10)descpritive analysis of month
summary(data$month)#tells it is catergorical variables
unique(data$month)#gives the unique entries
count(data,month)#cout of each occurrence
# Create a bar plot for 'Month'
ggplot(data = data, aes(x = month)) +
  geom_bar(fill = "#008080", color = "black") +
  labs(title = "Distribution of Month ",
       x = "Month ",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#11)descpritive analysis of day_of_week
summary(data$day_of_week)#tells it is catergorical variables
unique(data$day_of_week)#gives the unique entries
count(data,day_of_week)#cout of each occurrence
# Create a bar plot for 'day_of_week'
ggplot(data = data, aes(x = day_of_week)) +
  geom_bar(fill = "lightgray", color = "black") +
  labs(title = "Distribution of day of week ",
       x = "day of week ",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#12)descpritive analysis of poutcome
summary(data$poutcome)#tells it is catergorical variables
unique(data$poutcome)#gives the unique entries
count(data,poutcome)#cout of each occurrence
# Create a bar plot for 'poutcome'
ggplot(data = data, aes(x = poutcome)) +
  geom_bar(fill = "#008080", color = "black") +
  labs(title = "Distribution of Poutcome ",
       35
       x = "Poutcome ",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#13)descpritive analysis of subscribed
summary(data$subscribed)#tells it is catergorical variables
unique(data$subscribed)#gives the unique entries
count(data,subscribed)#cout of each occurrence
#pie chart for subscribed
ggplot(data = data, aes(x = "", fill = subscribed)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("#D3D3D3", "#008080")) + # Light grey and teal colors
  labs(title = "Subscription Status",
       fill = "Subscribed",
       x = NULL, y = NULL) +
  theme_void() +
  theme(legend.position = "bottom")
#14)descrptive analysis of contact_duration
summary(data$contact_duration)#gives mean mode median of the data
sum(is.na(data$contact_duration))
#plot of contact_duration
ggplot(data, aes(x = contact_duration)) +
  geom_histogram(binwidth = 100, fill = "#008080", color = "black") +
  labs(title = "Distribution of Contact Duration",
       x = "Contact Duration",
       y = "Frequency") +
  theme_minimal()
#15)descrptive analysis of campaign
summary(data$campaign)#gives mean mode median of the data
sum(is.na(data$campaign))
#plot of campaign
ggplot(data, aes(x = campaign)) +
  geom_histogram(binwidth = 1, fill = "lightgray", color = "black") +
  labs(title = "Distribution of Campaign",
       x = "Campaign",
       y = "Frequency") +
  theme_minimal()
#16)descrptive analysis of pdays
36
summary(data$pdays)#gives mean mode median of the data
sum(is.na(data$pdays))
#plot of days
ggplot(data = data, aes(x = pdays)) +
  geom_histogram(binwidth = 100, fill = "#008080", color = "black") +
  labs(title = "Distribution of Pdays",
       x = "Pdays",
       y = "Frequency") +
  theme_minimal()
#17)descrptive analysis of previous_contacts
summary(data$previous_contacts)#gives mean mode median of the data
sum(is.na(data$previous_contacts))
#hist of previous_contacts
ggplot(data = data, aes(x = previous_contacts)) +
  geom_histogram(binwidth = 1, fill = "lightgrey", color = "black") +
  labs(title = "Distribution of Previous Contacts",
       x = "Previous Contacts",
       y = "Frequency") +
  theme_minimal()
#18)descrptive analysis of previous_contacts
summary(data$emp_var_rate)#gives mean mode median of the data
sum(is.na(data$emp_var_rate))
#plot of emp_var_rate
ggplot(data = data, aes(x = emp_var_rate)) +
  geom_histogram(binwidth = 0.5, fill = "#008080", color = "black") +
  labs(title = "Distribution of Employment Variation Rate",
       x = "Employment Variation Rate",
       y = "Frequency") +
  theme_minimal()
#19)descrptive analysis of cons_price_idx
summary(data$cons_price_idx)#gives mean mode median of the data
sum(is.na(data$cons_price_idx))
#plot of emp_var_rate
ggplot(data, aes(x = cons_price_idx)) +
  geom_histogram(binwidth = 0.1, fill = "lightgrey", color = "black") +
  labs(title = "Distribution of Consumer Price Index",
       x = "Consumer Price Index",
       y = "Frequency") +
  theme_minimal()
37
#20)descrptive analysis of cons_conf_idx
summary(data$cons_conf_idx)#gives mean mode median of the data
sum(is.na(data$cons_conf_idx))
#plot of cons_conf_idx
ggplot(data = data, aes(x = cons_conf_idx)) +
  geom_histogram(binwidth = 1, fill = "#008080", color = "black") +
  labs(title = "Distribution of Consumer Confidence Index",
       x = "Consumer Confidence Index",
       y = "Frequency") +
  theme_minimal()
#21)descrptive analysis of Euribor_3m
summary(data$euribor_3m)#gives mean mode median of the data
sum(is.na(data$euribor_3m))
#plot of Euribor_3m
ggplot(data, aes(x = euribor_3m)) +
  geom_histogram(binwidth = 1, fill = "lightgrey", color = "black") +
  labs(title = "Distribution of Euribor 3 Month Rate",
       x = "Euribor 3 Month Rate",
       y = "Frequency") +
  theme_minimal()
#22)descrptive analysis of n_employed
summary(data$n_employed)#gives mean mode median of the data
sum(is.na(data$n_employed))
ggplot(data = data, aes(x = n_employed)) +
  geom_histogram(binwidth = 50, fill = "#008080", color = "black") +
  labs(title = "Distribution of Number of Employees",
       x = "Number of Employees",
       y = "Frequency") +
  theme_minimal()
#data cleaning
#1)age
attach(data)#attaching data varible for ease of typing
summary(age)#summmary of age
sum(is.na(age))#to find out if there are any null values
#box plot before data cleanning
ggplot(data)+
  38
geom_boxplot(aes(x=age),outlier.colour = "red")
# only one outlier 999 is found out
newdata <- data %>%
  filter(age !=999)
#box plot after data cleaning
ggplot(newdata)+
  geom_boxplot(aes(x=age),outlier.colour = "red")
#2)marital status
#summary statstics before cleaning data
summary(data$marital_status)
count(data,marital_status)
#There are 23 NA values need to replace them with the most occuring variable
newdata<-newdata %>%
  filter(!(is.na(marital_status)))
#summary statstics after cleaning data
summary(newdata$marital_status)
count(newdata,marital_status)
#barplot of variable marital_status
ggplot(newdata) +
  geom_bar(aes(x = marital_status, fill = marital_status))
#3)housing_loan
summary(newdata$housing_loan)#gives us the summary statstics
count(newdata,housing_loan)#gives the count of values of column housing_loan
sum(is.na(newdata$housing_loan))#gives us the sum of na if present
#barplot of variable housing_loan
ggplot(newdata) +
  geom_bar(aes(x = housing_loan, fill = housing_loan))
#4)occupation
summary(newdata$occupation)#gives us the summary statstics
count(newdata,occupation)#gives the count of values of column occupation
sum(is.na(newdata$occupation))#gives if na value is present
#barplot of variable occupation
ggplot(newdata) +
  geom_bar(aes(x = occupation, fill = occupation)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
39
#5)education level
summary(newdata$education_level)#gives us the summary statstics
count(newdata,education_level)#gives the count of values of column education level
sum(is.na(newdata$education_level))# tell us if there is na value present
#barplot of variable education level
ggplot(newdata) +
  geom_bar(aes(x = education_level, fill = education_level)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#6)personal loan
summary(newdata$personal_loan)#gives us the summary stastics
count(newdata,personal_loan)#gives unique values or count of column of personal_loan
sum(is.na(newdata$personal_loan))#tell us if there is na value present
#barplot of variable personal loan
ggplot(newdata) +
  geom_bar(aes(x = personal_loan, fill = personal_loan))
#7)month
summary(newdata$month)#gives us the summary statstics
count(newdata,month)#gives the count of values inside the col of month
# there are two jul and july we need to keep jul
sum(is.na(newdata$personal_loan))
#barplot of variable month
ggplot(newdata) +
  geom_bar(aes(x = month, fill = month))
#rename the july to jul as others are also july
newdata <- newdata %>%
  mutate(month = ifelse(month == "july", "jul", month))
summary(newdata$month)#gives us the summary statstics
count(newdata,month)#gives us the count of month
#8)days_of_week
summary(newdata$day_of_week)#gives us the summary stastics
count(newdata,day_of_week)#gives the count of values inside col of day_of_week
#here there is a two variables tue and tues we need change Tues to tue
sum(is.na(newdata$day_of_week))# tells if there is na values present
#rename tues to tue
newdata <- newdata %>%
  mutate(day_of_week = ifelse(day_of_week == "tues", "tue", day_of_week ))
summary(newdata$day_of_week)#gives us the summary stastics
40
count(newdata,day_of_week)#gives the count of values inside col of day_of_week
#barplot of day_of_week
ggplot(newdata) +
  geom_bar(aes(x = day_of_week, fill = day_of_week))
#9)campaign
summary(newdata$campaign)#gives us the summary stastics
count(newdata,campaign)#gives the count of values inside col of campaign
sum(is.na(newdata$campaign))#tells us if there is na values present
#box plot of campaign
ggplot(newdata)+
  geom_boxplot(aes(x=campaign),outlier.colour = "red")
#10)pdays
summary(newdata$pdays)#Gives us summary stats
count(newdata,pdays)#gives us the values inside pdays
sum(is.na(newdata$pdays))#tell is there any na values present
#boxplot of pdays
ggplot(newdata)+
  geom_boxplot(aes(x=pdays),outlier.colour = "red")
#11)poutcome
summary(newdata$poutcome)#gives us the summary stats
count(newdata,poutcome)#gives us the count of values in poutcome
sum(is.na(newdata$poutcome))#tells us is there is any na values
#barplot of poutcome
ggplot(newdata)+
  geom_bar(aes(x=poutcome,fill=poutcome))
#12) Credit_deafult
summary(newdata$credit_default)#gives us the summary stats
count(newdata,credit_default)#gives us the count of values in credit_default
sum(is.na(newdata$credit_default))#tells if there is any null vales
#barplot of credit_deafult
ggplot(newdata)+
  geom_bar(aes(x=credit_default,fill=credit_default))
#13)Contact_method
summary(newdata$contact_method)#gives us the summary stats
count(newdata,contact_method)#gives us the count of values in contact_method
sum(is.na(newdata$contact_duration))#tells if there is any null vales
41
#plot of contact method
ggplot(newdata)+
  geom_bar(aes(x=contact_method,fill=contact_method))
#14)n_employed
summary(newdata$n_employed)#gives us the summary stats
count(newdata,n_employed)#gives us the count of values in N_employed
sum(is.na(newdata$n_employed))#tell us if there is any null values
#plot of n_empployed
ggplot(newdata)+
  geom_histogram(aes(x=n_employed,bins=10,fill=n_employed))
#15)Euribor_3m
summary(newdata$euribor_3m)#gives us the summary stats
count(newdata,euribor_3m)#gives us the count of values in Euribor_3m
sum(is.na(newdata$euribor_3m))#tell us if there is any null values
#plot of Euribor_3m
ggplot(newdata)+
  geom_histogram(aes(x=euribor_3m))
#convert all as factor to numeric
newdata <- newdata %>%
  mutate_if(is.character,as.factor)
#Hypothesis Testing
#Personal loan, Pday, Occupation, Credit default, campaign
#1) Personal Loan
chisq.test(newdata$personal_loan, newdata$subscribed)
#Accepting the null hypothesis
# Create a contingency table
cont_table <- table(newdata$personal_loan, newdata$subscribed)
# Convert the contingency table to a dataframe for ggplot
df_for_plot <- as.data.frame(cont_table)
# Rename the columns for clarity
names(df_for_plot) <- c("PersonalLoan", "Subscribed", "Count")
# Create the bar plot
42
library(viridis)
# Define a color palette suitable for a white background
color_palette <- viridis_pal(option = "A", direction = 1)(3) # Adjust the number '3' based on
the number of categories
# Plot with the chosen color palette
ggplot(df_for_plot, aes(x = PersonalLoan, y = Count, fill = Subscribed)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Bar Plot of Personal Loan and Subscription Status",
       x = "Personal Loan",
       y = "Count") +
  scale_fill_manual(values = color_palette)
#2)pdays
t.test(pdays ~ subscribed, data = newdata)
#Rejecting the null hypothesis
#plot
ggplot(newdata, aes(x = subscribed, y = pdays, fill = subscribed)) +
  geom_boxplot(color = "black", fill = "#008080") + # Green color for the boxplot
  geom_point(position = position_jitterdodge(), alpha = 0.5, size = 2, color = "#800080") + #
  Purple color for the swarm plot
labs(x = "Subscribed", y = "Days Since Last Contact (pdays)",
     title = "Relationship between pdays and Subscription Status")
#extra violin graph
#appendix
ggplot(newdata, aes(x = subscribed, y = pdays, fill = subscribed)) +
  geom_violin() +
  labs(x = "Subscribed", y = "Days Since Last Contact (pdays)",
       title = "Relationship between pdays and Subscription Status")
#3) Occupation
chisq.test(newdata$occupation, newdata$subscribed)
ggplot(newdata, aes(x = occupation, fill = subscribed)) +
  geom_bar(position = "dodge") +
  labs(x = "Occupation", y = "Count", title = "Relationship between Occupation and
Subscription Status") +
  scale_fill_manual(values = c("#800080", "#008080"),
                    name = "Subscribed",
                    labels = c("No", "Yes")) +
  43
theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotating x-axis labels for better
readability
#Rejecting the null hypothesis
#4) Credit Default
chisq.test(newdata$credit_default, newdata$subscribed)
ggplot(newdata, aes(x = credit_default, fill = subscribed)) +
  geom_bar(position = "dodge", color = "black") +
  labs(x = "Credit Default", y = "Count", title = "Relationship between Credit Default and
Subscription Status") +
  scale_fill_manual(values = c("#800080", "#008080"),
                    name = "Subscribed",
                    labels = c("No", "Yes")) +
  theme_minimal()
# Rejecting the null hypothesis
#5) Campaign VS subscribed
t.test(campaign ~ subscribed, data = newdata)
# Assuming your dataset is named 'data'
library(ggplot2)
# Creating a boxplot with specified colors
ggplot(newdata, aes(x = subscribed, y = contact_duration, fill = subscribed)) +
  geom_boxplot() +
  labs(x = "Subscribed", y = "Contact Duration", title = "Relationship between Subscribed and
Contact Duration") +
  scale_fill_manual(values = c("#800080", "#008080"))
# Rejecting the null hypothesis
#model building
library(caret)
#set seed
set.seed(40412492)
#split the data set in training and test
index <- createDataPartition(newdata$subscribed,times = 1 ,p =0.8 ,list = FALSE)
44
train_data <- newdata[index,]
test_data <- newdata[-index,]
#Create to function to calculate R2
logisticPseudoR2s <- function(LogModel) {
  dev <- LogModel$deviance
  nullDev <- LogModel$null.deviance
  modelN <- length(LogModel$fitted.values)
  R.l <- 1 - dev / nullDev
  R.cs <- 1- exp ( -(nullDev - dev) / modelN)
  R.n <- R.cs / ( 1 - ( exp (-(nullDev / modelN))))
  cat("Pseudo R^2 for logistic regression\n")
  cat("Hosmer and Lemeshow R^2 ", round(R.l, 3), "\n")
  cat("Cox and Snell R^2 ", round(R.cs, 3), "\n")
  cat("Nagelkerke R^2 ", round(R.n, 3), "\n")
}
#1)using only hypothesis variables
model_1 <- glm(subscribed ~ pdays+occupation+credit_default+campaign,data =
                 train_data,family ="binomial")
#predictions of model 1
pred_1 <- predict(model_1,test_data,type = "response")
head(pred_1)
class_pred_hypo_1 <- as.factor(ifelse(pred_1 >.5 ,"yes","no"))
postResample(class_pred_hypo_1,test_data$subscribed)
confusionMatrix(data = class_pred_hypo_1,test_data$subscribed)
#to find the summary of model_1
summary(model_1)
#to find the R2 Model_1
logisticPseudoR2s(model_1)
#2)Hypothesis variables+ literature backed variables
model_2 <- glm(subscribed ~
                 pdays+occupation+credit_default+campaign+age+marital_status+education_level+day_of_
               week+month,data = train_data,family ="binomial")
pred_model_2 <- predict(model_2,test_data,type = "response")
head(pred_model_2)
class_pred_hypo_2 <- as.factor(ifelse(pred_model_2 >.5 ,"yes","no"))
postResample(class_pred_hypo_2,test_data$subscribed)
confusionMatrix(data = class_pred_hypo_2,test_data$subscribed)
45
#to find the summary of model_2
summary(model_2)
#to find the R2 Model_2
logisticPseudoR2s(model_2)
#3)All variables
model_3 <- glm(subscribed ~
                 pdays+occupation+credit_default+campaign+age+marital_status+education_level+day_of_
               week+month+poutcome+housing_loan+contact_duration+n_employed+euribor_3m,data =
                 train_data,family ="binomial")
pred_model_3 <- predict(model_3,test_data,type = "response")
head(pred_model_3)
class_pred_hypo_3 <- as.factor(ifelse(pred_model_3 >.5 ,"yes","no"))
postResample(class_pred_hypo_3,test_data$subscribed)
confusionMatrix(data = class_pred_hypo_3,test_data$subscribed)
#to find the summary of model_1
summary(model_3)
#to find the R2 Model_3
logisticPseudoR2s(model_3)
install.packages("stargazer")
library(stargazer)
stargazer(model_1, model_2, model_3,
          type = "html", title = "Logistic Regression Models Summary",
          out = "models_summary.html",
          column.labels = c("Model 1", "Model 2", "Model 3"),
          dep.var.caption = "Dependent Variable: Subscribed",
          model.numbers = FALSE,
          notes.label = "Significance Level",
          report = "vc*")
# View generated HTML file
browseURL("models_summary.html")
install.packages("pROC") # Install pROC package if not installed
library(pROC)
# Predict probabilities for test data
46
pred_probs <- predict(model_3, test_data, type = "response")
# Create ROC curve
roc_curve <- roc(test_data$subscribed, pred_probs)
# Plot ROC curve
plot(roc_curve, col = "blue", lwd = 2, print.auc = TRUE, print.auc.y = 0.2, print.auc.x = 0.7)
#plot model 3
plot(model_3) #ploting model graph
#Assumptions
library(car)
vif(model_3)