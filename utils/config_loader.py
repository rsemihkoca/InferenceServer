import yaml

def load_config(config_path):
    import os

    # Get the list of all files in the current directory
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print(files)

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)