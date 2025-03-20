import boto3
import csv
import glob
import os
import hashlib

# Helper function to generate a unique hash for a row (adjust as needed)
def generate_unique_id(row, file_name):
    unique_str = file_name + row['Pack Voltage'] + row['Pack Amperage (Current)'] + row['Pack State of Charge (SOC)']
    return hashlib.md5(unique_str.encode()).hexdigest()

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('Raw_Data_SeniorDesign')

csv_files = glob.glob("C:/Users/jmani/Documents/BatteryMLProject/src/data/four_files/*.csv")

for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    file_name = os.path.basename(csv_file)
    with open(csv_file, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # unique_id = generate_unique_id(row, file_name)
                item = {
                    # 'UniqueID': unique_id,  # This attribute must be defined as your table's partition key
                    'Pack_Voltage': row['Pack Voltage'],
                    'Pack_Amperage': row['Pack Amperage (Current)'],
                    'Pack_SOC': row['Pack State of Charge (SOC)'],
                    'Source_File': file_name
                }
                # Use a conditional write to avoid duplicates:
                table.put_item(
                    Item=item,
                    ConditionExpression='attribute_not_exists(UniqueID)'
                )
                print("Uploaded item:", item)
            except Exception as e:
                # Likely a duplicate if the condition failed; handle as needed
                print(f"Error uploading item {row} (possibly duplicate): {e}")
