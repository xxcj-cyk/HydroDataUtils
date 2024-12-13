import json

def serialize_json(my_dict, my_file):
    with open(my_file, "w") as FP:
        json.dump(my_dict, FP, indent=4)