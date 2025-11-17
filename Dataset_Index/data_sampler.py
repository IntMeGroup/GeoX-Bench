import json
import random
import os

def sample_json(input_filepath, output_filepath, sample_size):
    """
    Loads a JSON file, randomly samples `sample_size` entries (assuming the JSON data
    is a list of dictionaries), and writes the sampled data to another JSON file.

    Parameters:
        input_filepath (str): Path to the input JSON file.
        output_filepath (str): Path to the output JSON file.
        sample_size (int): Number of items to randomly sample.
    """
    # Load the JSON data from the input file
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # Check if the data is a list
    if not isinstance(data, list):
        raise ValueError("The JSON file does not contain a list of dictionaries.")
    
    # Check that sample_size is not larger than the available data
    if sample_size > len(data):
        raise ValueError(f"Sample size ({sample_size}) cannot be greater than the number of items ({len(data)}) in the JSON file.")
    
    # Randomly sample the data
    sampled_data = random.sample(data, sample_size)
    
    # Write the sampled data to the output file
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(sampled_data, outfile, indent=4)
    
    print(f"Successfully sampled {sample_size} items from {input_filepath} and saved them to {output_filepath}")

# Example usage
if __name__ == '__main__':
    input_file = "/gemini/code/LMMBench/json_data/LLMGeo_formatted_metadata_Location.json"           # Replace with your input JSON file path
    output_file = "LLMGeo_Location_13.json"    # Replace with your desired output JSON file path
    output_file = os.path.join('/gemini/code/LMMBench/Dataset_Index/100Sample',
                               output_file)
    x_num = 13
    
    sample_json(input_file, output_file, x_num)