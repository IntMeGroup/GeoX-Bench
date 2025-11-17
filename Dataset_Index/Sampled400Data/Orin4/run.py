#!/usr/bin/env python3
import json
import sys
import os

def inspect_json(filename):
    """
    Inspect a JSON file and count the number of dictionaries in a List[Dict] format.
    
    Args:
        filename (str): Path to the JSON file
    
    Returns:
        int: Number of dictionaries in the list
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            return None
        
        # Load and parse JSON file
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is a list
        if not isinstance(data, list):
            print(f"Error: JSON content is not a list. Found type: {type(data).__name__}")
            return None
        
        # Count dictionaries in the list
        dict_count = 0
        non_dict_count = 0
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                dict_count += 1
            else:
                non_dict_count += 1
                print(f"Warning: Item at index {i} is not a dict (type: {type(item).__name__})")
        
        return dict_count, non_dict_count
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{filename}': {e}")
        return None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect.py <json_file>")
        print("Example: python inspect.py data.json")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print(f"Inspecting: {filename}")
    print("-" * 40)
    
    result = inspect_json(filename)
    
    if result is not None:
        dict_count, non_dict_count = result
        total_items = dict_count + non_dict_count
        
        print(f"Total items in list: {total_items}")
        print(f"Number of dictionaries: {dict_count}")
        
        if non_dict_count > 0:
            print(f"Non-dict items: {non_dict_count}")
        
        if total_items > 0:
            percentage = (dict_count / total_items) * 100
            print(f"Dict percentage: {percentage:.1f}%")
        
        # Additional info
        if dict_count > 0:
            print(f"\n✓ Valid List[Dict] format with {dict_count} dictionaries")
        else:
            print(f"\n✗ No dictionaries found in the list")

if __name__ == "__main__":
    main()