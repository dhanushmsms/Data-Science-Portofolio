#Import required libraries
library(tidyverse)
library(dplyr)
#get the working directory
getwd()
#set the working directory
setwd("/Users/dhanush/Desktop/Bussiness analytics /Data management ")
#Give alias to tables
data1<- Data_1_Customer
data2<- Data_2_Motor_Policies
data3<- Data_3_Health_Policies
data4<- Data_4_Travel_Policies
#Join tables using dpylr
ABT <- data1 %>%
  left_join(data2,by = "MotorID") %>%
  left_join(data3,by ="HealthID") %>%
  left_join(data4,by="TravelID")
#data quality analysis
#summarising the data
summary(ABT)
str(ABT)
#Fiding out data quality issues and outlier
count(ABT, CustomerID)
count(ABT, Title) # Mr and Mr. are one and the same and Mr must be renaned to Mr.
count(ABT,GivenName)
count(ABT,MiddleInitial)
count(ABT, Surname)
count(ABT, CardType)# There is a outlier in cardtype which says 0
count(ABT, Occupation)
count(ABT, Gender)#There is an outiler with f , m which must be changed to male and
female
count(ABT, Age)
#ggplot age
ggplot(ABT)+
  geom_boxplot(aes(x=Age),outlier.color = "red",outlier.size=2)#we can find that there are 3
outlier in data
summary(ABT$Age)#summarising age
count(ABT, Location)
count(ABT, MotorID)
count(ABT, HealthID)
count(ABT, TravelID)
count(ABT,MotorType)
count(ABT,veh_value)
count(ABT,Exposure)
count(ABT, clm)
count(ABT,Numclaims)
count(ABT,v_body)
count(ABT,v_ae)
count(ABT,HealthType)
count(ABT,HealthDependentsAdults)
count(ABT,DependentsKids)#there is an outlier 40
#ggplot of DepednentKids
ggplot(ABT)+
  geom_boxplot(aes(x=DependentsKids),outlier.color = "blue",outlier.size=2)#we can find
that there are 3 outlier in data
summary(ABT$DependentsKids)
count(ABT, TravelType)
count(ABT, ComChannel)#there outliters for email,phone, sms like e , p and s which must
recoded as email phone and sms
is.na(ABT)#finding out any null values
count(ABT, ComChannel)
#Adressing data quality issues mentioned above
ABT_clean <- ABT %>%
  mutate(Title = ifelse(Title == "Mr","Mr.",Title)) %>%
  mutate(CardType=ifelse(CardType == 0,"Not_Given",CardType)) %>%
  mutate(Gender=ifelse(Gender=="f","female",Gender)) %>%
  mutate(Gender=ifelse(Gender=="m","male",Gender)) %>%
  mutate(ComChannel=ifelse(ComChannel=="E","Email",ComChannel)) %>%
  mutate(ComChannel=ifelse(ComChannel=="P","Phone",ComChannel)) %>%
  mutate(ComChannel=ifelse(ComChannel=="S","SMS",ComChannel)) %>%
  filter(!(Age<18 | Age >85))
#replace value "40" Depedentkids with mean 2
ABT_clean$DependentsKids <- replace(ABT_clean$DependentsKids,
                                    ABT_clean$DependentsKids>4,2)
#ggplot age after cleaning
ggplot(ABT_clean)+
  geom_boxplot(aes(x=Age),outlier.color = "red",outlier.size=2)
#ggplot Dependentskids after cleaning
ggplot(ABT_clean)+
  geom_boxplot(aes(x=DependentsKids),outlier.color = "blue",outlier.size=2)
#Exporting clean ABT data
install.packages("openxlsx")
library(openxlsx)
write.xlsx(ABT_clean,"ABT_clean.xlsx")