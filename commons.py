import random

flat = {
    "name": "flat",
    "params": ["amp1", "amp2", "mod16", "mod66", "mod11", "mod23", "mod34", "mod45", "ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio6", "detune1", "detune2", "detune3", "detune4", "detune5", "detune6", "feedback", "lfofreq", "lfodepth"],
    "values_range": {
        "mod": [float(e)/1000 for e in range(0, 5000, 100)],
        "ratio": [float(e)/1000 for e in range(0, 200, 2)],
        "feedback": [float(e)/100 for e in range(100, 202, 2)],
        "lfofreq": [float(e)/100 for e in range(0, 100, 2)],
    },
    "constants": {
        "amp1": 1,
        "amp2": 1,
        "lfodepth": 0,
        "detune1": 0, 
        "detune2": 0, 
        "detune3": 0, 
        "detune4": 0, 
        "detune5": 0,
        "detune6": 0,
    },
    "PATCHES_IN_TOTAL": 9000,
    "SAMPLE_LENGTH": 0.5,
}

wave = {
    "name": "wave",
    "params": ["amp1", "amp2", "mod16", "mod66", "mod11", "mod23", "mod34", "mod45", "ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio6", "detune1", "detune2", "detune3", "detune4", "detune5", "detune6", "feedback", "lfofreq", "lfodepth", "egrate11", "egrate21", "egrate31", "egrate41", "egrate51", "egrate61", "egrate12","egrate22", "egrate32", "egrate42", "egrate52", "egrate62", "egrate13", "egrate23", "egrate33", "egrate43", "egrate53", "egrate63", "egrate14", "egrate24", "egrate34", "egrate44", "egrate54", "egrate64", "egrate14", "egrate24", "egrate34", "egrate44", "egrate54", "egrate64", "eglevel14", "eglevel24", "eglevel34", "eglevel44", "eglevel54", "eglevel64", "sustain"],
    "values_range": {
        "mod": [float(e)/1000 for e in range(0, 1000, 50)], # 1000 is the bound, after that ugly noise starts to get in (at least when using this conf)
        "ratio": [float(e)/1000 for e in range(0, 200, 2)],
        "feedback": [float(e)/100 for e in range(100, 202, 2)],
        "lfofreq": [float(e)/100 for e in range(0, 100, 2)],
    },
    "constants": {
        "amp1": 1,
        "amp2": 1,
        "lfodepth": 0,
        "detune1": 0, 
        "detune2": 0, 
        "detune3": 0, 
        "detune4": 0, 
        "detune5": 0,
        "detune6": 0,
        "egrate11": 0.2,
        "egrate21": 0.2,
        "egrate31": 0.2,
        "egrate41": 0.2,
        "egrate51": 0.2,
        "egrate61": 0.2,
        "egrate12": 0,
        "egrate22": 0,
        "egrate32": 0,
        "egrate42": 0,
        "egrate52": 0,
        "egrate62": 0,
        "egrate13": 0,
        "egrate23": 0,
        "egrate33": 0,
        "egrate43": 0,
        "egrate53": 0,
        "egrate63": 0,
        "egrate14": 0.2,
        "egrate24": 0.2,
        "egrate34": 0.2,
        "egrate44": 0.2,
        "egrate54": 0.2,
        "egrate64": 0.2,
        "eglevel14": 0,
        "eglevel24": 0,
        "eglevel34": 0,
        "eglevel44": 0,
        "eglevel54": 0,
        "eglevel64": 0,
        "sustain": 4
    },
    "PATCHES_IN_TOTAL": 8000,
    "SAMPLE_LENGTH": 4,
}


def sample_params(context):
    return {
        param: sampling_strategy(context, param) if get_value_range(context, param) else context["constants"][param] for param in context["params"]
    }

def sampling_strategy(context, param):
    value_range = get_value_range(context, param)
    param_value = random.choice(value_range)
    return param_value

def get_value_range(context, param):
    for k, v in context["values_range"].items():
        if k in param:
            return v
