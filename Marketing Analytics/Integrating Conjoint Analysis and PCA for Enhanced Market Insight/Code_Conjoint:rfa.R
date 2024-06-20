Code for PCA

# Marketing analysis 

#PCA (Principal Componenet Analysis)


#loading librarires 
library(tidyverse)
library(data.table)

#set seed
set.seed(40412492)

#set working directory 
getwd()
setwd('/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Marketing Analytics/Assigment_2')

#load the pca dataset 
pca_data <- read_csv('PCA data.csv')

#Remove column one i.e model
pca_data <- pca_data[,-1]

# Standardize the data
data_scaled <- scale(pca_data)

# Perform PCA
pca <- prcomp(data_scaled, retx = TRUE, scale. = TRUE)

# Summary of PCA to see the proportion of variance explained
print(summary(pca))

# Calculate and view loading factors
loadings <- pca$rotation
print(loadings)

# Calculate singular values
singular_values <- pca$sdev
print(singular_values)

# Proportion of Variance Explained
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
print(var_explained)

# Plotting a perceptual map
perceptual_map_data <- as.data.frame(pca$x[, 1:2])
colnames(perceptual_map_data) <- c("PC1", "PC2")
ggplot(perceptual_map_data, aes(x = PC1, y = PC2)) +
  geom_point() +
  xlab("Principal Component 1") +
  ylab("Principal Component 2") +
  ggtitle("Perceptual Map") +
  theme_minimal()

#Important Feature selection 
fviz_cos2(pca,choice = "var",axes = 1:2)


fviz_eig(pca, addlabels = TRUE)
#or

# Adding a biplot to visualize both scores and loadings
biplot(pca, scale = 0)  # scale = 0 to keep arrows and points in proportion

# Using fviz_pca_var to visualize variables' contribution
fviz_pca_var(pca, col.var = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)  # Avoid text overlapping

Code for Conjoint Analysis 

#Conjoint analysis 

#setting up the attrib level

#setting seed

set.seed(40412492)

#loading library conjonint
library(conjoint)

#creating list of attributes and list

attrib.level <- list(
  Environmental_friendliness = c("0% CO2 reduction", "30% CO2 reduction", "50% CO2 reduction"),
  Delivery_time = c("14 days", "21 days", "30 days"),
  Service_level = c("5-year warranty", "5-year warranty & free maintenance", 
                    "5-year warranty, free maintenance and installation, & upgradeability"),
  Price = c("1000 GBP", "1200 GBP", "1500 GBP"),
  Quality_of_material = c("Market average", "A bit higher than market average"),
  Marketing_proficiency = c("Not very proficient and poor communication", "Very proficient and have good communication skills")
)

## Create the fractional factorial design
#Top 18 product profiles 
experiment <- expand.grid(attrib.level)
design <- caFactorialDesign(data=experiment, type="fractional", cards=22, seed=40412492) #(optimal)

## Check for correlation in fractional factorial design
#Answers the first question 
print(cor(caEncodedDesign(design)))
print(experiment)

#uploading data for pref and profiles 
attrib.vector <- data.frame(unlist(attrib.level,use.names=FALSE))

#Load Conjoint Prefernces.xlsx
pref <- readxl::read_excel("/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Marketing Analytics/Assigment_2/Conjoint Prefernces.xlsx")
pref <- pref[,2:11]

# Design - Load Product Profiles.csv
design <- read.csv("/Users/dhanush/Desktop/Bussiness analytics /Sem 2/Marketing Analytics/Assigment_2/Product Profiles.csv")
design <- design[,2:7]

#Calculation of part-worths
caPartUtilities(y = pref, x = design, z = unlist(attrib.level))
part_worths <- caPartUtilities(y = pref, x = design, z = unlist(attrib.level))

attrib.vector <- data.frame(unlist(attrib.level,use.names=FALSE))
colnames(attrib.vector) <- c("levels")
part.worths <- NULL
for (i in 1:ncol(pref)){
  temp <- caPartUtilities(pref[,i], design, attrib.vector)
  ## Pick the baseline case
  ## Adjust coding as needed based on number of attributes and levels
  ## Base Case: Environmental friendliness - 0% CO2 reduction; Delivery time (order fulfilment time) - 14 days;
  ## Service level - 5-year warranty ; Price - 1000 GBP ; Quality of material -  Market average; Marketing proficiency - Not very proficient and poor communication 
  
  Base_Enviromental <- temp[,"0% CO2 reduction"]; Base_Delivery <- temp[,"14 days"]; Base_Service <- temp[,"5-year warranty"]
  Base_Price <- temp[,"1000 GBP"]; Base_Quality <- temp[,"Market average"]; Base_Marketing <- temp[,"Not very proficient and poor communication"]
  ## Adjust Intercept
  temp[,"intercept"] <- temp[,"intercept"] - Base_Enviromental - Base_Marketing - Base_Quality - Base_Price - Base_Service - Base_Delivery
  ## Adjust Coefficients
  ## Envirnomental
  L1 <- length(attrib.level$Environmental_friendliness) + 1 ## Add 1 for the intercept
  for (j in 2:L1){temp[,j] <- temp[,j] - Base_Enviromental}
  ## Delivery
  L2 <- length(attrib.level$Delivery_time) + L1
  for (k in (L1+1):L2){temp[,k] <- temp[,k] - Base_Delivery}
  ## Service
  L3 <- length(attrib.level$Service_level) + L2
  for (l in (L2+1):L3){temp[,l] <- temp[,l] - Base_Service}
  ## Price
  L4 <- length(attrib.level$Price) + L3
  for (m in (L3+1):L4){temp[,m] <- temp[,m] - Base_Price}
  ## Quality
  L5 <- length(attrib.level$Quality_of_material) + L4
  for (n in (L4+1):L5){temp[,n] <- temp[,n] - Base_Quality}
  ## Marketing
  L6 <- length(attrib.level$Marketing_proficiency) + L5
  for (n in (L5+1):L6){temp[,n] <- temp[,n] - Base_Marketing}
  
  part.worths <- rbind(part.worths, temp)
}

rownames(part.worths) <- colnames(pref)

write.csv(part.worths, file.choose(new=TRUE), row.names = FALSE)

#Do cojoint analysis on given files 
Conjoint(pref, design, attrib.vector)

cluster <- caSegmentation(y=pref, x=design, c=3)
plot(fviz_cluster(cluster$segm, scale(cluster$util)))
