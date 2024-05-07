import json

def count_rouge_greater_than_zero(json_file):
    # Initialize count
    count = 0
    tot=0
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Iterate through each object in the array
    for obj in data:
        # Check if the "rouge" key exists and its value is greater than 0
        if "rouge" in obj and obj["rouge"] > 0:
            count += 1
            print(obj)
        tot+=1
    
    return tot,count

# Example usage
json_file = "./predicted.json"  # Replace with the path to your JSON file
tot,num_objects_with_rouge_gt_zero = count_rouge_greater_than_zero(json_file)
print("Number of objects with rouge greater than 0:", num_objects_with_rouge_gt_zero)
print("number of objects:",tot)
