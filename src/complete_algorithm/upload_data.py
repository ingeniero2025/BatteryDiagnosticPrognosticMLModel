import boto3
import csv
import glob
import os

# Initialize the DynamoDB resource and specify the table
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('Raw_Data_SeniorDesign')

# Use glob to find all CSV files in the specified folder
csv_files = glob.glob("C:/Users/jmani/Documents/BatteryMLProject/src/data/four_files/*.csv")

def upload_files():
    """Upload CSV files to DynamoDB table."""
    # Check if any CSV files were found
    # Process each CSV file found
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        file_name = os.path.basename(csv_file)  # Get the file name for the 'Source_File' attribute
        with open(csv_file, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            # Optional: use batch_writer for improved performance
            with table.batch_writer() as batch:
                for row in reader:
                    try:
                        # Convert data types if necessary (uncomment if needed)
                        # voltage = float(row['Pack Voltage'])
                        # amperage = float(row['Pack Amperage (Current)'])
                        # soc = int(row['Pack State of Charge (SOC)'])
                        item = {
                            'Pack_Voltage': row['Pack Voltage'],
                            'Pack_Amperage': row['Pack Amperage (Current)'],
                            'Pack_SOC': row['Pack State of Charge (SOC)'],
                            'Source_File': file_name
                        }
                        batch.put_item(Item=item)
                        print("Uploaded item:", item)
                    except Exception as e:
                        print(f"Error uploading item {row}: {e}")

def view_table_contents():
    """Retrieve and print all items from the DynamoDB table."""
    response = table.scan()
    items = response.get('Items', [])
    for item in items:
        print(item)

