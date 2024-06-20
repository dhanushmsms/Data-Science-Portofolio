SQL code :
. Code for left join in SQL
SELECT customer.CustomerID, customer.Title, customer.GivenName,
customer.MiddleInitial, customer.Surname, customer.CardType, customer.Gender,
customer.Occupation, customer.Age, customer.Location, customer.ComChannel,
customer.MotorID, motor_policies.PolicyStart, motor_policies.PolicyEnd,
motor_policies.MotorType, motor_policies.veh_value, motor_policies.Exposure,
motor_policies.clm, motor_policies.Numclaims, motor_policies.v_body,
motor_policies.v_age, motor_policies.LastClaimDate, health_policies.policyStart,
health_policies.policyEnd, health_policies.HealthType,
health_policies.HealthDependentsAdults, health_policies.DependentsKids,
travel_policies.policyStart, travel_policies.PolicyEnd, travel_policies.TravelType
INTO ABT
FROM ((customer LEFT JOIN motor_policies ON customer.[MotorID] =
motor_policies.[MotorID]) LEFT JOIN health_policies ON customer.[HealthID] =
health_policies.[HealthID]) LEFT JOIN travel_policies ON customer.[TravelID] =
travel_policies.[TravelID];
2. Descriptive Analysis using SQL
SELECT Gender, COUNT(*) AS Frequency
FROM ABT
GROUP BY Gender;
SELECT ABT.DependentsKids, Count(*) AS [COUNT]
FROM ABT
GROUP BY ABT.DependentsKids;
SELECT MAX(Age) AS MaxAge, MIN(Age) AS MinAge
FROM ABT;
SELECT ComChannel, COUNT(*) AS Total_count
FROM ABT
GROUP BY ComChannel;
SELECT CardType, COUNT(*) AS Frequency
FROM ABT
GROUP BY CardType;
SELECT Title, COUNT(*) AS [Count]
FROM ABT
GROUP BY Title;
3. Insights Queries using SQL
Retention rate
SELECT 'Motor' AS PolicyType,
SUM(IIF(DATEDIFF('d', PolicyStart, PolicyEnd) >= 365, 1, 0)) AS RenewedPolicies,
COUNT(*) AS TotalPolicies,
(SUM(IIF(DATEDIFF('d', PolicyStart, PolicyEnd) >= 365, 1, 0)) / COUNT(*)) * 100 AS
RetentionRate
FROM ABT_clean
WHERE MotorID IS NOT NULL
UNION ALL
SELECT 'Health' AS PolicyType,
SUM(IIF(DATEDIFF('d', policyStart, policyEnd) >= 365, 1, 0)) AS RenewedPolicies,
COUNT(*) AS TotalPolicies,
(SUM(IIF(DATEDIFF('d', policyStart, policyEnd) >= 365, 1, 0)) / COUNT(*)) * 100 AS
RetentionRate
FROM ABT_clean
WHERE HealthID IS NOT NULL
UNION ALL
SELECT 'Travel' AS PolicyType,
SUM(IIF(DATEDIFF('d', policyStart, PolicyEnd) >= 365, 1, 0)) AS RenewedPolicies,
COUNT(*) AS TotalPolicies,
(SUM(IIF(DATEDIFF('d', policyStart, PolicyEnd) >= 365, 1, 0)) / COUNT(*)) * 100 AS
RetentionRate
FROM ABT_clean
WHERE TravelID IS NOT NULL;
Communication VS Location
SELECT ComChannel, Location, COUNT(*) AS CustomerCount
FROM ABT_clean
GROUP BY ComChannel, Location;
Location VS Gender VS Policy Count
SELECT Location, Gender, Count(*) AS PolicyCount
FROM ABT_clean
GROUP BY Location, Gender;
Age-Group VS Total_Claims
SELECT
Gender,
IIF(Age BETWEEN 18 AND 25, '18-25',
IIF(Age BETWEEN 26 AND 40, '26-40',
IIF(Age BETWEEN 41 AND 55, '41-55',
IIF(Age BETWEEN 56 AND 60, '56-60',
IIF(Age > 60, 'Above 60', 'Other')
)
)
)
) AS age_range,
COUNT(CustomerID) AS total_customers,
COUNT(IIF(HealthID IS NOT NULL, customerID, NULL)) AS health_policies,
COUNT(IIF(MotorID IS NOT NULL, customerID, NULL)) AS motor_policies,
COUNT(IIF(TravelID IS NOT NULL, customerID, NULL)) AS travel_policies
FROM
ABT_clean
GROUP BY
Gender,
IIF(age BETWEEN 18 AND 25, '18-25',
IIF(age BETWEEN 26 AND 40, '26-40',
IIF(age BETWEEN 41 AND 55, '41-55',
IIF(age BETWEEN 56 AND 60, '56-60',
IIF(age > 60, 'Above 60', 'Other')
)
)
)
);
Age_group VS Claim %
SELECT
IIF(ABT_clean.Age BETWEEN 18 AND 25, '18-25 Entry Level / Early Professional',
IIF(ABT_clean.Age BETWEEN 26 AND 40, '26-40 Mid-Career / Intermediate
Professional',
IIF(ABT_clean.Age BETWEEN 41 AND 55, '41-55 Senior/Experienced
Professional',
IIF(ABT_clean.Age BETWEEN 56 AND 60, '56-60 Pre-Retirement/Late
Career', 'Other')
)
)
) AS AgeGroup,
COUNT(*) AS CustomerCount,
ROUND(( AVG(ABT_clean.Numclaims)/MAX(ABT_clean.Numclaims))*100,2) AS
AvgClaimspercentage
FROM ABT_clean
GROUP BY
IIF(ABT_clean.Age BETWEEN 18 AND 25, '18-25 Entry Level / Early Professional',
IIF(ABT_clean.Age BETWEEN 26 AND 40, '26-40 Mid-Career / Intermediate
Professional',
IIF(ABT_clean.Age BETWEEN 41 AND 55, '41-55 Senior/Experienced
Professional',
IIF(ABT_clean.Age BETWEEN 56 AND 60, '56-60 Pre-Retirement/Late
Career', 'Other')
)
)
);
Gender VS Diff Policies under diff Age_Range
SELECT
Gender,
IIF(Age BETWEEN 18 AND 25, '18-25',
IIF(Age BETWEEN 26 AND 40, '26-40',
IIF(Age BETWEEN 41 AND 55, '41-55',
IIF(Age BETWEEN 56 AND 60, '56-60',
IIF(Age > 60, 'Above 60', 'Other')
)
)
)
) AS age_range,
COUNT(CustomerID) AS total_customers,
COUNT(IIF(HealthID IS NOT NULL, customerID, NULL)) AS health_policies,
COUNT(IIF(MotorID IS NOT NULL, customerID, NULL)) AS motor_policies,
COUNT(IIF(TravelID IS NOT NULL, customerID, NULL)) AS travel_policies
FROM
ABT_clean
GROUP BY
Gender,
IIF(age BETWEEN 18 AND 25, '18-25',
IIF(age BETWEEN 26 AND 40, '26-40',
IIF(age BETWEEN 41 AND 55, '41-55',
IIF(age BETWEEN 56 AND 60, '56-60',
IIF(age > 60, 'Above 60', 'Other')
)
)
)
);