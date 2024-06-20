#probllem 3 Make-to-Stock Chemotherapy Drugs
# Load necessary libraries for modeling and solving MIP
library(ompr)           # For modeling
library(ompr.roi)       # For solving the model using ROI (R Optimization Infrastructure)
library(ROI.plugin.glpk) # GLPK (GNU Linear Programming Kit) solver plugin

# Model setup
model <- MIPModel() %>%
  # Decision variables: quantities of EU and US base constituents for Chemo1 and Chemo2
  add_variable(x1EU, type = "integer", lb = 0) %>%
  add_variable(x1US, type = "integer", lb = 0) %>%
  add_variable(x2EU, type = "integer", lb = 0) %>%
  add_variable(x2US, type = "integer", lb = 0) %>%
  
  # Objective: Maximize profit calculated as revenue minus cost for both drugs
  set_objective(1200 * (x1EU + x1US) + 1400 * (x2EU + x2US) - 800 * x1EU - 1500 * x1US - 800 * x2EU - 1500 * x2US, "max") %>%
  
  # Constraints
  # D-metric constraints for Chemo1 and Chemo2 
  add_constraint(25*x1EU + 15*x1US <= 23 * (x1EU + x1US)) %>%
  add_constraint(25*x2EU + 15*x2US <= 23 * (x2EU + x2US)) %>%
  
  # P-metric constraints for Chemo1 and Chemo2 
  add_constraint(87*x1EU + 98*x1US >= 88 * (x1EU + x1US)) %>%
  add_constraint(87*x2EU + 98*x2US >= 93 * (x2EU + x2US)) %>%
  
  # Demand constraints to meet market requirements
  add_constraint(x1EU + x1US >= 100000) %>%
  add_constraint(x1EU + x1US <= 200000) %>%
  add_constraint(x2EU + x2US >= 10000) %>%
  add_constraint(x2EU + x2US <= 40000) %>%
  
  # Supply constraints based on available base constituents
  add_constraint(x1EU + x2EU <= 80000) %>%
  add_constraint(x1US + x2US <= 120000)

# Solve the model using the GLPK solver
solution <- solve_model(model, with_ROI(solver = "glpk"))

# Extract solution values for decision variables
x1EU_sol <- get_solution(solution, x1EU)
x1US_sol <- get_solution(solution, x1US)
x2EU_sol <- get_solution(solution, x2EU)
x2US_sol <- get_solution(solution, x2US)

# Calculate D-metrics and P-metrics for Chemo1 and Chemo2 based on solution
D_metric_Chemo1 <- (25*x1EU_sol + 15*x1US_sol) / (x1EU_sol + x1US_sol)
D_metric_Chemo2 <- (25*x2EU_sol + 15*x2US_sol) / (x2EU_sol + x2US_sol)
P_metric_Chemo1 <- (87*x1EU_sol + 98*x1US_sol) / (x1EU_sol + x1US_sol)
P_metric_Chemo2 <- (87*x2EU_sol + 98*x2US_sol) / (x2EU_sol + x2US_sol)

# Output the results: optimal quantities, profit, and drug metrics
cat("Quantities to blend for Chemo1: EU =", x1EU_sol, ", US =", x1US_sol, "\n")
cat("Quantities to blend for Chemo2: EU =", x2EU_sol, ", US =", x2US_sol, "\n")
cat("Maximum monthly profit: €", solution$objective_value, "\n")
cat("D-metrics: Chemo1 =", D_metric_Chemo1, ", Chemo2 =", D_metric_Chemo2, "\n")
cat("P-metrics: Chemo1 =", P_metric_Chemo1, ", Chemo2 =", P_metric_Chemo2, "\n")
