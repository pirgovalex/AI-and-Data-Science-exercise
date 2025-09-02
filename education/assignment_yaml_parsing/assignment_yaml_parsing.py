import yaml
import argparse
import os


def take_input()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="YAML parsing assignment")
    parser.add_argument('-i', '--input', required=True, help='Path to the intended YAML file')
    parser.add_argument('-k', '--key', required=True, help='Key from the YAML file to be checked')
    args = parser.parse_args()
    args.input = path_validation(args.input)
    args.key = string_validation(args.key)
    return args
def string_validation(string:str)->str:
    #validate input
    if not isinstance(string,str):
        raise ValueError("Input must be a string")
    if not string.strip():
        raise ValueError("Input string cannot be empty")
    return string
def path_validation(user_path:str)->str:
    #validate path
    if not os.path.exists(user_path) :
        raise FileNotFoundError(f"Path {user_path} does not exist")
    elif not user_path.lower().endswith((".yaml", ".yml")):
        raise ValueError(f"File is not of the .yaml format")
    return os.path.abspath(user_path)
def check_key_in_yaml(data: dict, key: str):
    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):  # dict
            if k in current:
                current = current[k]
            else:
                raise KeyError(f"Key '{key}' not found in the YAML file")
        elif isinstance(current, list):  # list lvl
            try:
                idx = int(k)
            except ValueError:
                raise KeyError(f"Expected list index at '{k}' in key '{key}'")
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                raise IndexError(f"List index {idx} out of range for key '{key}'")
        else:
            raise KeyError(f"Cannot go deeper at '{k}' — reached non-iterable value.")
    return current
def main() -> None:
    args = take_input()
    try:
        with open(args.input, 'r') as file:
            data = yaml.safe_load(file) or {}
            data = dict(data)
        result = check_key_in_yaml(data, args.key)
        print(f"Key '{args.key}' found in the YAML file with value: {result}")
    except FileNotFoundError as e:
        print(f"File error: {e}") #Bbad practice to print in production code, but okay for this example
    except KeyError as e:
        print(f"Key error: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
if __name__ == "__main__":
    main()
