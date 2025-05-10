import random
import csv
import os

# Define the number of records to generate
num_records = 1000

# Define column names based on the original dataset
columns = [
    'PatientAge', 'BodyMassIndex', 'BloodGlucose', 'SystolicBP', 
    'InsulinLevel', 'SkinFold', 'FamilyHistory', 'ActivityLevel', 'DiabetesStatus'
]

# Create a list to store the data
data = []

# Define value ranges and special cases
age_range = (18, 85)
bmi_range = (16.0, 45.0)
blood_glucose_range = (70, 300)
systolic_bp_range = (60, 200)
insulin_range = (0, 300)
skin_fold_range = (10, 50)
family_history_range = (0.0, 1.0)
activity_level_range = (0, 5)

# Function to introduce empty values, extreme values, and broken values
def introduce_special_cases(value, probability=0.05, extreme_probability=0.02, broken_probability=0.01):
    # Empty value
    if random.random() < probability:
        return ""
    
    # Extreme value (very high or very low)
    if random.random() < extreme_probability:
        if isinstance(value, (int, float)) and value != 0:
            return value * random.choice([5, 10, -1, 0.01])
    
    # Broken value (non-numeric for numeric fields)
    if random.random() < broken_probability:
        return random.choice(["N/A", "error", "unknown", "#VALUE!"])
    
    return value

# Generate data
for i in range(num_records):
    # Generate base values
    age = random.randint(*age_range)
    bmi = round(random.uniform(*bmi_range), 1)
    blood_glucose = random.randint(*blood_glucose_range)
    systolic_bp = random.randint(*systolic_bp_range)
    insulin = random.randint(*insulin_range)
    skin_fold = random.randint(*skin_fold_range)
    family_history = round(random.uniform(*family_history_range), 2)
    activity_level = random.randint(*activity_level_range)
    
    # Diabetes status is more likely to be 1 if blood glucose is high
    diabetes_prob = 0.1 + (blood_glucose - blood_glucose_range[0]) / (blood_glucose_range[1] - blood_glucose_range[0]) * 0.8
    diabetes_status = 1 if random.random() < diabetes_prob else 0
    
    # Apply special cases
    age = introduce_special_cases(age)
    bmi = introduce_special_cases(bmi)
    blood_glucose = introduce_special_cases(blood_glucose)
    systolic_bp = introduce_special_cases(systolic_bp)
    insulin = introduce_special_cases(insulin)
    skin_fold = introduce_special_cases(skin_fold)
    family_history = introduce_special_cases(family_history)
    activity_level = introduce_special_cases(activity_level)
    diabetes_status = introduce_special_cases(diabetes_status, probability=0.03)
    
    # Add row to data
    row = [age, bmi, blood_glucose, systolic_bp, insulin, skin_fold, family_history, activity_level, diabetes_status]
    data.append(row)

# Add some unique/special cases
# 1. Add a few rows with extremely high BMI
for _ in range(5):
    idx = random.randint(0, num_records-1)
    if isinstance(data[idx][1], (int, float)):
        data[idx][1] = round(random.uniform(50, 100), 1)

# 2. Add a few rows with extremely high blood glucose
for _ in range(5):
    idx = random.randint(0, num_records-1)
    if isinstance(data[idx][2], (int, float)):
        data[idx][2] = random.randint(400, 600)

# 3. Add a few rows with extremely high blood pressure
for _ in range(5):
    idx = random.randint(0, num_records-1)
    if isinstance(data[idx][3], (int, float)):
        data[idx][3] = random.randint(220, 300)

# 4. Add a few rows with negative values (which shouldn't exist in medical data)
for _ in range(10):
    idx = random.randint(0, num_records-1)
    col = random.randint(1, 5)  # Choose a random numeric column
    if isinstance(data[idx][col], (int, float)):
        data[idx][col] = -1 * data[idx][col]

# Write to CSV file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

output_file = os.path.join(data_dir, 'large_synthetic_diabetes.csv')
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(data)

print(f"Generated {num_records} records in {output_file}")
