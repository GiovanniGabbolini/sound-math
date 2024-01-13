from commons import *
from supercollider import Server, Synth
import time
import json
from tkinter import *
import pyqtgraph as pg
import os

server = Server()

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
    composition = json.load(open("composition.json"))
    composition2 = json.load(open("composition2.json"))

    try:

        for step, step2 in zip(composition, composition2):
            
            patch = json.load(open(f"batch/patches/{step}.json"))
            patch2 = json.load(open(f"batch/patches/{step2}.json"))

            update_visuals(figure, patch)
            synth = Synth(server, "superfm", patch)
            synth2 = Synth(server, "superfm", patch2)

            time.sleep(0.5)
            synth.free()
            synth2.free()

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
    explore()
