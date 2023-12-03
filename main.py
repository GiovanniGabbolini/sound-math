from supercollider import Server, Synth
import time
import random
import json
from tkinter import *
import pyqtgraph as pg
import os
from threading import Thread

server = Server()

params = ["amp1", "amp2", "mod16", "mod66", "mod11", "mod23", "mod34", "mod45", "ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio6", "detune1", "detune2", "detune3", "detune4", "detune5", "detune6", "feedback", "lfofreq", "lfodepth",
# "eglevel11", "eglevel12", "eglevel13", "eglevel14", "eglevel21", "eglevel22", "eglevel23", "eglevel24", "eglevel31", "eglevel32", "eglevel33", "eglevel34", "eglevel41", "eglevel42", "eglevel43", "eglevel44", "eglevel51", "eglevel52", "eglevel53", "eglevel54", "eglevel61", "eglevel62", "eglevel63", "eglevel64", "egrate11", "egrate12", "egrate13", "egrate14", "egrate21", "egrate22", "egrate23", "egrate24", "egrate31", "egrate32", "egrate33", "egrate34", "egrate41", "egrate42", "egrate43", "egrate44", "egrate51", "egrate52", "egrate53", "egrate54", "egrate61", "egrate62", "egrate63", "egrate64",
]

values_range = {
    "mod": [float(e)/1000 for e in range(0, 5000, 100)],
    "ratio": [float(e)/1000 for e in range(0, 200, 2)],
    "feedback": [float(e)/100 for e in range(100, 202, 2)],
    "lfofreq": [float(e)/100 for e in range(0, 100, 2)],
}

print(values_range)

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
    # "eglevel11":1,
    # "eglevel12":0.5,
    # "eglevel13":0.5,
    # "eglevel14":0,
    # "eglevel21":1,
    # "eglevel22":0.5,
    # "eglevel23":0.5,
    # "eglevel24":0,
    # "eglevel31":1,
    # "eglevel32":0.5,
    # "eglevel33":0.5,
    # "eglevel34":0,
    # "eglevel41":1,
    # "eglevel42":0.5,
    # "eglevel43":0.5,
    # "eglevel44":0,
    # "eglevel51":1,
    # "eglevel52":0.5,
    # "eglevel53":0.5,
    # "eglevel54":0,
    # "eglevel61":1,
    # "eglevel62":0.5,
    # "eglevel63":0.5,
    # "eglevel64":0,
    # "egrate11":0.1,
    # "egrate12":0.2,
    # "egrate13":0.2,
    # "egrate14":0.1,
    # "egrate21":0.1,
    # "egrate22":0.2,
    # "egrate23":0.2,
    # "egrate24":0.1,
    # "egrate31":0.1,
    # "egrate32":0.2,
    # "egrate33":0.2,
    # "egrate34":0.1,
    # "egrate41":0.1,
    # "egrate42":0.2,
    # "egrate43":0.2,
    # "egrate44":0.1,
    # "egrate51":0.1,
    # "egrate52":0.2,
    # "egrate53":0.2,
    # "egrate54":0.1,
    # "egrate61":0.1,
    # "egrate62":0.2,
    # "egrate63":0.2,
    # "egrate64":0.1,
}

def get_value_range(param):
    for k, v in values_range.items():
        if k in param:
            return v

def sampling_strategy(param):
    value_range = get_value_range(param)
    param_value = random.choice(value_range)
    return param_value

def sample_params():
    return {
        k: sampling_strategy(k) if get_value_range(k) else constants[k] for k in params
    }

def step_forward(param, value):
    value_range = get_value_range(param)
    current_position = value_range.index(value)
    next_position = value_range[current_position+1]
    return next_position

def step_backward(param, value):
    value_range = get_value_range(param)
    current_position = value_range.index(value)
    next_position = value_range[current_position-1]
    return next_position

def step(current_patch, new_patch):
    for k in current_patch.keys():
        if current_patch[k] != new_patch[k]:
            
            if current_patch[k] < new_patch[k]:
                current_patch[k] = step_forward(k, current_patch[k])
            else:
                current_patch[k] = step_backward(k, current_patch[k])

            return current_patch

def patch_inferface_for_plot(current_patch):
    normalise = lambda value, min, max: (float(value)-float(min))/(float(max)-float(min))

    xs = []
    ys = []

    for param, value in current_patch.items():
        if get_value_range(param):
            min, max = (get_value_range(param)[0], get_value_range(param)[-1])
            
            xs.append(normalise(value, min, max))
            ys.append(len(ys))

    return xs, ys

def init_visuals():
    figure = pg.plot()
    return figure

def update_visuals(figure, params):
    xs, ys = patch_inferface_for_plot(params)
    figure.plot(xs, ys, clear=True)
    pg.QtWidgets.QApplication.processEvents()

def live():
    figure = init_visuals()

    current_patch = sample_params()
    new_patch = current_patch

    try:
        while True:
            for _ in range(2, 1000):

                if current_patch == new_patch:
                    new_patch = sample_params()

                current_patch = step(current_patch, new_patch)
                update_visuals(figure, current_patch)

                synth = Synth(server, "superfm", current_patch)
                print("Created synth")
                print("Frequency: %.1f" % synth.get("freq"))
                time.sleep(2)
                synth.free()

    except KeyboardInterrupt:
        synth.free()
        print("Freed synth")

def explore():
    params_to_widget = {}

    multipliers = {
        "mod": 1000,
        "ratio": 1000,
        "feedback": 100,
        "lfofreq": 1000,
        "lfodepth": 10,
    }

    def get_multiplier(param):
        for k, v in multipliers.items():
            if k in param:
                return v
        return 1
    
    def save_patch():
        saved_patches = os.listdir("python/patches")
        latest_patch_index = sorted([int(e.split(".")[0]) for e in saved_patches])[-1]
        new_patch_index = latest_patch_index + 1
        with open(f"python/patches/{new_patch_index}.json", "w") as f:
            json.dump(current_patch, f, indent=4)

    def send_to_synth(event):
        for param in params:
            if param not in constants:
                current_patch[param]=params_to_widget[param].get()/float(get_multiplier(param))
        synth = Synth(server, "superfm", current_patch)
        time.sleep(0.5)
        synth.free()

    master = Tk()
    
    current_patch = sample_params()
    for i_param, param in enumerate(sorted(params)):

        if not get_value_range(param):
            continue

        multiplier = get_multiplier(param)
        scale = Scale(
            master,
            from_=float(-5000),
            to=float(5000),
            orient=HORIZONTAL,
            label=f"{param}*{multiplier}",
            width=30,
            length=400,
        )
        scale.set(current_patch[param]*multiplier)
        scale.bind("<ButtonRelease-1>", send_to_synth)
        scale.grid(row=i_param%10, column=i_param//10)

        params_to_widget[param]=scale

    button = Button(master, command=save_patch)
    button.grid(row=(i_param+1)%10, column=(i_param+1)//10)

    mainloop()


def debug():
    try:
        while True:
            params = json.load(open("python/default_params.json"))

            synth = Synth(server, "superfm", params)
            time.sleep(0.1)
            synth.free()
    
    except KeyboardInterrupt:
        synth.free()


if __name__ == "__main__":
    live()
