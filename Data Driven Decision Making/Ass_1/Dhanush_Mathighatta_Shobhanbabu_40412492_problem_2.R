#AI Chip - Problem 2 

#We make use of library Ompr to solve the problem 
#As the required libraries are already installed we call the libraries using the syntax shown below
# Utilizing the 'ompr' library to optimize production allocation
# Load the required packages for optimization
library(ompr)
library(ompr.roi)
library(ROI.plugin.glpk) # Ensure this solver is installed

# Parameters 
P_i <- c(1950, 1850, 2000, 1800) # Selling price per chip for each customer
C_j <- c(A = 1150, B = 1250) # Cost to make each chip at fab
S_ij <- matrix(c(300, 400, 550, 450, 600, 300, 400, 250), nrow = 2, byrow = TRUE) # Cost to deliver chips
prod_capacity <- c(A = 50, B = 42) # Production capacities in millions
demand <- c(36, 46, 11, 35) # Maximum demand for each customer


# Model building for question 2 
model <- MIPModel() %>%
  add_variable(x[i, j], i = 1:4, j = 1:2, type = "integer", lb = 0) %>%
  set_objective(sum_expr((P_i[i] - C_j[j] - S_ij[j, i]) * x[i, j], i = 1:4, j = 1:2), "max") %>%
  add_constraint(sum_expr(x[i, 1], i = 1:4) <= prod_capacity["A"], j = 1) %>%
  add_constraint(sum_expr(x[i, 2], i = 1:4) <= prod_capacity["B"], j = 2) %>%
  add_constraint(sum_expr(x[i, j], j = 1:2) <= demand[i], i = 1:4) %>%
  solve_model(with_ROI(solver = "glpk", verbose = TRUE))



# Solution 
solution <- get_solution(model, x[i, j])
cat("Solution:\n")
print(solution)
model















