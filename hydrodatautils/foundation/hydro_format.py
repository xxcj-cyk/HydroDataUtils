import json
from pathlib import Path

def serialize_json(my_dict, my_file):
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_paths(item) for item in obj]
        return obj
    
    with open(my_file, "w") as FP:
        json.dump(convert_paths(my_dict), FP, indent=4)


def unserialize_json(my_file):
    with open(my_file, "r") as fp:
        my_object = json.load(fp)
    return my_object