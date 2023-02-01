import json
import pickle
import tqdm as _tqdm
import yaml


def tqdm(x, **kwargs):
    return _tqdm.tqdm(x, disable=logger.tqdm_disable, **kwargs)


def warn_unparsed(_dict, parsed_keys, config_file):
    diff = set(_dict.keys()).difference(set(parsed_keys))
    if diff:
        logger.warning(f"Unparsed {diff} in file {config_file}")


def raise_none_field(name, file):
    raise ValueError("{} is None in file {}".format(name, file))


def read_yaml(file):
    with open(file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_yaml(data, file):
    with open(file, "w+") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def read_json(file):
    with open(file, "r") as f:
        lines = [json.loads(line) for line in f]
    if len(lines) == 1:
        return lines[0]
    return lines


def write_json(data, file):
    if not isinstance(data, list):
        data = [data]
    with open(file, "w+") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def read_pickle(file):
    _df = pickle.load(open(file, "rb"))
    return _df


def write_pickle(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=4)