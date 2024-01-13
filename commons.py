import random

params = ["amp1", "amp2", "mod16", "mod66", "mod11", "mod23", "mod34", "mod45", "ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio6", "detune1", "detune2", "detune3", "detune4", "detune5", "detune6", "feedback", "lfofreq", "lfodepth",]

values_range = {
    "mod": [float(e)/1000 for e in range(0, 5000, 100)],
    "ratio": [float(e)/1000 for e in range(0, 200, 2)],
    "feedback": [float(e)/100 for e in range(100, 202, 2)],
    "lfofreq": [float(e)/100 for e in range(0, 100, 2)],
}

constants = {
    "amp1": 1,
    "amp2": 1,
    "lfodepth": 0,
    "detune1": 0, 
    "detune2": 0, 
    "detune3": 0, 
    "detune4": 0, 
    "detune5": 0,
    "detune6": 0,
}

def sample_params():
    return {
        k: sampling_strategy(k) if get_value_range(k) else constants[k] for k in params
    }

def sampling_strategy(param):
    value_range = get_value_range(param)
    param_value = random.choice(value_range)
    return param_value

def get_value_range(param):
    for k, v in values_range.items():
        if k in param:
            return v
