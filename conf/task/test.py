import os
import yaml
import glob
from pathlib import Path
import argparse
import shutil
from typing import Any, Dict, List, Union

def transform_paths(data: Any, path_mappings: Dict[str, str]) -> Any:
    """
    Recursively transform paths in YAML data structure.
    
    Args:
        data: The data structure to transform (dict, list, or primitive)
        path_mappings: Dictionary mapping old paths to new paths
    
    Returns:
        Transformed data structure
    """
    if isinstance(data, dict):
        # Transform each value in the dictionary
        return {key: transform_paths(value, path_mappings) for key, value in data.items()}
    
    elif isinstance(data, list):
        # Transform each item in the list
        return [transform_paths(item, path_mappings) for item in data]
    
    elif isinstance(data, str):
        # Apply path transformations to string values
        result = data
        # Apply transformations in order (more specific paths first)
        for old_path, new_path in path_mappings.items():
            result = result.replace(old_path, new_path)
        return result
    
    else:
        # Return primitive values unchanged
        return data

def process_yaml_files(folder_path: str, output_folder: str = None, backup: bool = True):
    """
    Process all YAML files in a folder and transform specified paths.
    
    Args:
        folder_path: Path to folder containing YAML files
        output_folder: Optional output folder (if None, files are modified in place)
        backup: Whether to create backup files before modification
    """
    
    # Define path mappings (order matters - more specific first)
    path_mappings = {
        '/home2/ys_zheng/LMMBench': '/home/LMMBench',
        '/home2/ys_zheng': '/home/Data'
    }
    
    # Ensure folder path exists
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Setup output folder
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = folder_path
    
    # Find all YAML files
    yaml_patterns = ['*.yaml', '*.yml']
    yaml_files = []
    for pattern in yaml_patterns:
        yaml_files.extend(folder_path.glob(pattern))
    
    if not yaml_files:
        print(f"No YAML files found in {folder_path}")
        return
    
    print(f"Found {len(yaml_files)} YAML files to process...")
    
    # Process each YAML file
    for yaml_file in yaml_files:
        try:
            print(f"Processing: {yaml_file.name}")
            
            # Read the YAML file
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Transform paths in the data
            transformed_data = transform_paths(data, path_mappings)
            
            # Determine output file path
            if output_folder:
                output_file = output_path / yaml_file.name
            else:
                output_file = yaml_file
                
                # Create backup if requested and modifying in place
                if backup:
                    backup_file = yaml_file.with_suffix(yaml_file.suffix + '.backup')
                    shutil.copy2(yaml_file, backup_file)
                    print(f"  Created backup: {backup_file.name}")
            
            # Write the transformed YAML
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(transformed_data, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False, indent=2)
            
            print(f"  Transformed and saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {yaml_file.name}: {str(e)}")

def preview_changes(folder_path: str, max_files: int = 5):
    """
    Preview the changes that would be made to YAML files without actually modifying them.
    
    Args:
        folder_path: Path to folder containing YAML files
        max_files: Maximum number of files to preview
    """
    
    path_mappings = {
        '/home2/ys_zheng/LMMBench': '/home/LMMBench',
        '/home2/ys_zheng': '/home/Data'
    }
    
    folder_path = Path(folder_path)
    yaml_patterns = ['*.yaml', '*.yml']
    yaml_files = []
    for pattern in yaml_patterns:
        yaml_files.extend(folder_path.glob(pattern))
    
    print(f"Preview of changes for up to {min(max_files, len(yaml_files))} files:")
    print("=" * 60)
    
    for i, yaml_file in enumerate(yaml_files[:max_files]):
        print(f"\nFile: {yaml_file.name}")
        print("-" * 40)
        
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Show what changes would be made
            changes_found = False
            for old_path, new_path in path_mappings.items():
                if old_path in content:
                    changes_found = True
                    count = content.count(old_path)
                    print(f"  Would replace {count} occurrence(s) of:")
                    print(f"    '{old_path}' -> '{new_path}'")
            
            if not changes_found:
                print("  No changes needed")
                
        except Exception as e:
            print(f"  Error reading file: {str(e)}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Transform paths in YAML files')
    parser.add_argument('folder', help='Folder containing YAML files')
    parser.add_argument('-o', '--output', help='Output folder (optional)')
    parser.add_argument('--no-backup', action='store_true', 
                       help='Skip creating backup files when modifying in place')
    parser.add_argument('--preview', action='store_true', 
                       help='Preview changes without modifying files')
    
    args = parser.parse_args()
    
    try:
        if args.preview:
            preview_changes(args.folder)
        else:
            process_yaml_files(
                folder_path=args.folder,
                output_folder=args.output,
                backup=not args.no_backup
            )
            print("\nProcessing complete!")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) == 1:
        # Interactive mode if no arguments provided
        print("YAML Path Transformer")
        print("=" * 30)
        
        folder_path = input("Enter folder path containing YAML files: ").strip()
        
        preview = input("Preview changes first? (y/n): ").strip().lower() == 'y'
        if preview:
            preview_changes(folder_path)
            
            proceed = input("\nProceed with transformation? (y/n): ").strip().lower() == 'y'
            if not proceed:
                print("Cancelled.")
                sys.exit(0)
        
        output_folder = input("Output folder (leave empty to modify in place): ").strip()
        if not output_folder:
            output_folder = None
            
        backup = input("Create backup files? (y/n): ").strip().lower() != 'n'
        
        process_yaml_files(folder_path, output_folder, backup)
        print("\nProcessing complete!")
    else:
        # Command line mode
        main()
