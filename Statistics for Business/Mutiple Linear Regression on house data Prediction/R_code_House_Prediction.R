library(tidyverse)
library(dplyr)
#to know the working directory
getwd()
#set the working directory
setwd("/Users/dhanush/Desktop/Bussiness analytics /Stastics for bussiness ")
#import the data set
library(readxl)
data <- read_excel("ames.xlsx")
colnames(data)
summary(data$ID)
unique(data)
class(data$ID)
summary(data$d_type)
summary(data$zone)
summary(data$lot_area)
summary(data$road)
summary(data$house_quality)
summary(data$house_condition)
summary(data$year_built)
summary(data$year_remod)
summary(data$veneer_area)
summary(data$bsmt_sf1)
summary(data$bsmt_sf2)
summary(data$bsmt_unf)
summary(data$bsmt_area)
summary(data$aircon)
summary(data$floor1_sf)
summary(data$floor2_sf)
summary(data$low_qual_sf)
summary(data$bsmt_full_bath)
summary(data$bsmt_half_bath)
summary(data$full_bath)
summary(data$half_bath)
summary(data$bedroom)
summary(data$kitchen)
summary(data$rooms_tot)
summary(data$fireplace)
summary(data$garage_cars)
summary(data$garage_area)
summary(data$deck_sf)
summary(data$open_porch_sf)
summary(data$encl_porch_sf)
summary(data$season_porch)
summary(data$screen_porch)
summary(data$pool_sf)
summary(data$features_val)
summary(data$month_sold)
summary(data$year_sold)
summary(data$sale_price)
#Data cleaning
#1)lot_area
#cleaning of lot area
summary(lot_area)
ggplot(data , aes(x = lot_area)) +
  geom_boxplot(fill = "black", outlier.color = "red") # Color outliers in red
# IOQ = Q3-Q1
#lower bound = Q1 - 1.5 * IQR
#upper boound = Q3 + 1.5 * IQR
new_data <- data %>% filter(lot_area >= 1271.5 & lot_area <=17753.5 )
#ggplot after cleaning the data
ggplot(new_data, aes(x = lot_area)) +
  geom_boxplot(fill = "black", outlier.color = "red") # Color outliers in red
#2)zone
#cleaning of zone
summary(data$zone)
count(data,zone)
sum(is.na(data$zone))
#convert nominal data into factor
new_data$zone <- as.factor(new_data$zone)
summary(new_data$zone)
ggplot(new_data)+
  geom_bar(aes(x=zone,fill="red"))
# 3) Neighbourd
summary(data$neighbourhood)
count(data,neighbourhood)
sum(is.na(data$neighbourhood))
#convert nominal data into factor
new_data$neighbourhood <- as.factor(new_data$neighbourhood)
summary(new_data$neighbourhood)
# 4) frontage
summary(data$frontage)
# again summarise data without including NA
summary(data$frontage, na.rm = T)
mean(data$frontage, na.rm = T)
median(data$frontage, na.rm= T)
ggplot(data, aes(x = frontage)) +
  geom_boxplot(fill = "black", outlier.color = "red") # Color outliers in red
ggplot(data, aes(x = lot_area)) +
  geom_histogram(fill = "black", outlier.color = "red") # Color outliers in red
#replace NA with mean
new_data <- data %>%
  mutate(frontage = ifelse(is.na(frontage), 68, frontage))
summary(new_data$frontage)
ggplot(new_data, aes(x = frontage)) +
  geom_boxplot(fill = "black", outlier.color = "red") # Color outliers in red
#remove outliers
new_data <- data %>% filter(frontage >= 32.625 & frontage <= 105.625 )
#ggplot after rermoving outliers
ggplot(new_data, aes(x = frontage)) +
  geom_boxplot(fill = "black", outlier.color = "red") # Color outliers in red
# 5)year_built
summary(data$year_built)
sum(is.na(data$year_built))
ggplot(new_data, aes(x = year_built)) +
  geom_boxplot(fill = "black", outlier.color = "red") # Color outliers in red
# as we have already cleaned some data in above it has already filtered some outliers
# 6) cleaning of year_remod
summary(year_remod)
sum(is.na(year_remod))
#ggplot using clean data of shown above
ggplot(new_data, aes(x = year_remod)) +
  geom_boxplot(fill = "black", outlier.color = "red")
#7) half_bath
summary(data$half_bath)
count(data,half_bath)
sum(is.na(half_bath))
#ggplot using clean data of shown above
ggplot(new_data, aes(x = half_bath)) +
  geom_boxplot(fill = "black", outlier.color = "red")
#8) full_bath
summary(data$full_bath)
count(data,full_bath)
sum(is.na(full_bath))
class(full_bath)
#ggplot using clean data of shown above
ggplot(new_data, aes(x = full_bath)) +
  geom_boxplot(fill = "black", outlier.color = "red")
#9)bedroom
summary(data$bedroom)
count(data,bedroom)
sum(is.na(data$bedroom))
#ggplot using clean data of shown above
ggplot(new_data, aes(x = bedroom)) +
  geom_boxplot(fill = "black", outlier.color = "red")
#10) kitchen
summary(data$kitchen)
count(data,kitchen)
sum(is.na(data$kitchen))
#ggplot using clean data of shown above
ggplot(new_data, aes(x = kitchen)) +
  geom_boxplot(fill = "black", outlier.color = "red")
#11) foundations
summary(data$foundations)
count(data,foundations)
sum(is.na(data$foundations))
#convert data in to factor
new_data$foundations <- as.factor(new_data$foundations)
summary(new_data$foundations)
count(new_data,foundations)
sum(is.na(new_data$foundations))
barchart(new_data$foundations)
# 12) stories
summary(data$stories)
count(data,stories)
sum(is.na(data$foundations))
#convert data in to factor
new_data$stories <- as.factor(new_data$stories)
summary(new_data$stories)
count(new_data,stories)
sum(is.na(new_data$stories))
#plotting to check if converted to factor
barchart(new_data$stories)
#13) room_ tot
summary(data$rooms_tot)
count(data,rooms_tot)
sum(is.na(data$rooms_tot))
#plot box plot to check outliers
ggplot(new_data, aes(x = rooms_tot)) +
  geom_boxplot(fill = "black", outlier.color = "red")
#keep outliers as u can back those
#14) Aircon
summary(data$aircon)
count(data,aircon)
sum(is.na(data$aircon))
#convert this into factor
new_data$aircon <- as.factor(new_data$aircon)
#to check if converetd into factor
summary(new_data$aircon)
count(new_data,aircon)
sum(is.na(new_data$aircon))
barchart(new_data$aircon)
#15) Heat_type
summary(data$heat_type)
count(data,heat_type)
sum(is.na(data$heat_type))
#convert to factor
new_data$heat_type <- as.factor(new_data$heat_qual)
summary(new_data$heat_type)
count(new_data,heat_type)
sum(is.na(new_data$heat_type))
barchart(new_data$heat_type)
#16) house_quality
summary(data$house_quality)
count(new_data,house_quality)
sum(is.na(data$house_quality))
new_data <- new_data %>%
  filter(house_quality <11)
count(data_1,house_quality)
hist(data_1$house_quality)
#convert to nomial data as mentioned in the data dictonary
new_data$house_quality <- as.factor(new_data$house_quality)
barchart(new_data$house_quality)
#17) House condtion
summary(data$house_condition)
count(new_data,house_condition)
sum(is.na(data$house_condition))
#convert to ordinal data
new_data$house_condition <- as.factor(new_data$house_condition)
barchart(new_data$house_condition)
#Convert all character into factor
new_data<-new_data %>% mutate_if(is.character,as.factor)
#Hypothesis testing
#H1 - lot_area vs sale_price
cor.test(new_data$sale_price, new_data$lot_area, method = "pearson")
m1_lot_area <- lm(new_data$sale_price ~ new_data$lot_area)
summary(m1_lot_area)
#plotting relationship between lot_area vs sale_price
ggplot(new_data, aes(x = lot_area, y = sale_price, color = sale_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Lot Area", y = "Sale Price") +
  ggtitle("Relationship between Lot Area and Sale Price") +
  scale_color_gradient(low = "blue", high = "red")
#H2 - Neighbourhood vs sale_price
m2_neighbourhood <- lm(new_data$sale_price ~ new_data$neighbourhood)
summary(m2_neighbourhood)
anova_neighbourhood <- anova(m2_neighbourhood)
summary(anova_neighbourhood)
print(anova_neighbourhood)
#plotting relationship between Neighbourhood vs sale_price
ggplot(hq1, aes(x = neighbourhood, y = avg_sale_price, fill = avg_sale_price)) +
  geom_bar(stat = "identity") + # Creating bars
  labs(x = "Neighbourhood", y = "Average Sale Price", title = " Average Sale Price by
Neighbourhood") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # Rotating x-axis labels
  scale_fill_gradient(low = "blue", high = "red")
#H3 room_tot vs sale_price
cor.test(new_data$sale_price, new_data$rooms_tot, method = "pearson")
m3_room_tot <- lm(new_data$sale_price ~ new_data$rooms_tot)
summary(m3_room_tot)
#create a scatter plot for the same
ggplot(new_data, aes(x = rooms_tot, y = sale_price, color = sale_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "rooms_tot", y = "Sale Price") +
  ggtitle("Relationship between rooms_tot and Sale Price") +
  scale_color_gradient(low = "blue", high = "red")
#H4 year of remod
cor.test(new_data$sale_price, new_data$year_remod, method = "pearson")
m4_year_remod <- lm(new_data$sale_price ~ new_data$year_remod)
summary(m4_year_remod)
#create a scatter plot for the same
ggplot(new_data, aes(x = year_remod, y = sale_price, color = sale_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "year_remod", y = "Sale Price") +
  ggtitle("Relationship between year_remod and Sale Price") +
  scale_color_gradient(low = "blue", high = "red")
#H5 frontage vs sale_price
cor.test(new_data$sale_price, new_data$frontage, method = "pearson")
m5_frontage <- lm(new_data$sale_price ~ new_data$frontage)
summary(m5_frontage)
#plottig sactter plot
ggplot(new_data, aes(x = frontage, y = sale_price, color = sale_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "frontage", y = "Sale Price") +
  ggtitle("Relationship between frontage and Sale Price") +
  scale_color_gradient(low = "blue", high = "red")
class(new_data$house_quality)
hq <- new_data %>% group_by(house_quality) %>% summarise(avg_sale_price =
                                                           mean(sale_price))
hq
ggplot(hq)+
  geom_bar(aes(x = house_quality, y = avg_sale_price, fill = house_quality), stat = "identity")
hq1 <- new_data %>% group_by(neighbourhood) %>% summarise(avg_sale_price =
                                                            mean(sale_price))
hq1
#divide the clean data into train and test
library(caret)
set.seed(40412492)
index <-createDataPartition(new_data$sale_price,times = 1 , p=0.7 , list = F)
train_data <- new_data[index,]
test_data <- new_data[-index,]
summary(test_data$neighbourhood)
#model testing
#model building using forward approcah
#model1 using hpyothesis vairables
model_1 <-lm(sale_price ~ lot_area + neighbourhood + rooms_tot + year_remod +
               frontage,data = train_data)
summary(hypothesis_based_model)
predictions_hypo_model <- predict(hypothesis_based_model,newdata = test_data)
postResample(predictions_hypo_model,test_data$sale_price)
#model2 using literature backed variables+hypo
model_2 <-lm(sale_price ~lot_area + neighbourhood + rooms_tot + year_remod + frontage+
               year_built + zone + half_bath + full_bath + bedroom + aircon,data = train_data)
summary(lit_based_model)
predictions_lit_model <- predict(lit_based_model,newdata = test_data)
postResample(predictions_lit_model,test_data$sale_price)
#model 3 including all variables
model_3 <- lm(sale_price ~ lot_area+ zone + neighbourhood + frontage + year_built +
                year_remod+half_bath+full_bath+bedroom+kitchen+foundations+stories+rooms_tot+airco
              n+heat_type+house_quality+house_condition,data = train_data)
summary(all_var_model)
predictions_all_var_model <- predict(all_var_model,newdata = test_data)
postResample(predictions_all_var_model,test_data$sale_price)
plot(all_var_model)
plot(model_3, which = 2)
install.packages("lmtest")
library(lmtest)
install.packages('car')
library(car)
vif(all_var_model)
plot(all_var_model)
dwtest(all_var_model)