# Statistical Hypothesis Testing
# 1. Z-Test
# 2. Independent t-Test
# Application: Student Performance Analysis
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import ztest

# Z-TEST
print("----- Z-Test -----")
# Sample marks of students
sample_marks = np.array([78, 82, 85, 88, 90, 76, 95, 89, 84, 91])
# Population mean
population_mean = 80
# Perform one-sample Z-test
z_stat, p_value = ztest(sample_marks, value=population_mean)
print("Sample Marks:")
print(sample_marks)
print("\nPopulation Mean:", population_mean)
print("\nZ-Statistic:", z_stat)
print("P-Value:", p_value)
# Decision
alpha = 0.05
if p_value < alpha:
    print("\nResult: Reject Null Hypothesis")
else:
    print("\nResult: Fail to Reject Null Hypothesis")
# INDEPENDENT t-TEST
print("\n\n----- Independent t-Test -----")
# Marks of two different classes
class_A = np.array([78, 82, 85, 88, 90])
class_B = np.array([72, 75, 70, 68, 74])
# Perform independent t-test
t_stat, p_value = stats.ttest_ind(class_A, class_B)
print("Class A Marks:")
print(class_A)
print("\nClass B Marks:")
print(class_B)
print("\nt-Statistic:", t_stat)
print("P-Value:", p_value)
# Decision
if p_value < alpha:
    print("\nResult: Reject Null Hypothesis")
else:
    print("\nResult: Fail to Reject Null Hypothesis")