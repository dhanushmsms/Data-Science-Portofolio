#ApTx-nova - Problem - 1
#We make use of library Ompr to solve the problem 
#As the required libraries are already installed we call the libraries using the syntax shown below
# Utilizing the 'ompr' library to optimize production allocation
# Load the required packages for optimization
getwd()
library(tidyverse)
library(ompr) # Used for modeling the optimization problem
library(ompr.roi) # Connects 'ompr' models to the 'ROI' optimization infrastructure
library(ROI.plugin.glpk) # Enables the use of the GLPK solver through 'ROI'

# Define the optimization model
model <- MIPModel() %>%
  # Introduce binary decision variables x[i,j] where i is the product and j is the plant
  add_variable(x[i, j], i = 1:4, j = 1:5, type = "binary") %>%
  # Define the objective function to maximize the total number of batches
  set_objective(1200*x[1,1] + 1400*x[1,2] + 600*x[1,3] + 800*x[1,4] + 800*x[1,5] +
                  0*x[2,1] + 1200*x[2,2] + 0*x[2,3] + 0*x[2,4] + 1400*x[2,5] +
                  600*x[3,1] + 800*x[3,2] + 200*x[3,3] + 0*x[3,4] + 1000*x[3,5] +
                  1000*x[4,1] + 1000*x[4,2] + 600*x[4,3] + 1200*x[4,4] + 1600*x[4,5], "max") %>%
  # Adding constraints to ensure each product is made by only one plant
  add_constraint(sum_expr(x[1,j], j = 1:5) == 1) %>%
  add_constraint(sum_expr(x[2,j], j = c(2,5)) == 1) %>%
  add_constraint(sum_expr(x[3,j], j = c(1,2,3,5)) == 1) %>%
  add_constraint(sum_expr(x[4,j], j = 1:5) == 1) %>%
  # Adding constraints to ensure that each plant produces at most one product
  add_constraint(sum_expr(x[i,1], i = 1:4) <= 1) %>%
  add_constraint(sum_expr(x[i,2], i = 1:4) <= 1) %>%
  add_constraint(sum_expr(x[i,3], i = 1:4) <= 1) %>%
  add_constraint(sum_expr(x[i,4], i = 1:4) <= 1) %>%
  add_constraint(sum_expr(x[i,5], i = 1:4) <= 1) %>%
  # Solve the model using GLPK solver and display the output
  solve_model(with_ROI(solver = "glpk", verbose = TRUE))

# Extracting the solution and displaying it
solution <- get_solution(model, x[i,j])
print(solution) # Show the optimal production allocation
model # Output a summary of the model

