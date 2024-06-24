from commons import *
from scipy.io.wavfile import write
from IPython.display import Audio
import sounddevice as REC
import numpy as np
from supercollider import Server, Synth
from scipy.io import wavfile
from math import ceil, floor
import json
import librosa
from tqdm import tqdm


def sample_space(context):
    server = Server()

    # Recording properties
    SAMPLE_RATE = 44100
    SECONDS = context["SAMPLE_LENGTH"]
    # Recording device
    REC.default.device = 'Soundflower (2ch)'

    i = 0
    while i < context["PATCHES_IN_TOTAL"]:
        patch = sample_params(context)
        synth = Synth(server, "superfm", patch)

        recording = REC.rec( int(SECONDS * SAMPLE_RATE), samplerate = SAMPLE_RATE, channels = 1)
        REC.wait()

        write(f"batch/audio_{context['name']}/{i}.wav", SAMPLE_RATE, recording)
        json.dump(patch, open(f"batch/patches_{context['name']}/{i}.json", "w"))

        synth.free()

        i += 1

def compute_similarity(context):
    SR = 44100

    SR, examplar = wavfile.read(f"batch/audio_{context['name']}/{0}.wav")
    patches = np.zeros((context["PATCHES_IN_TOTAL"], len(examplar)), dtype=np.float32)

    for i in tqdm(range(context["PATCHES_IN_TOTAL"])):
        _, data = wavfile.read(f"batch/audio_{context['name']}/{i}.wav") # load the data
        patches[i]=data
    patches = (patches.T - np.mean(patches, axis=1)).T # subtract mean

    examplar = librosa.feature.mfcc(y=examplar, sr=SR).flatten()
    mffcs = np.zeros((context["PATCHES_IN_TOTAL"], len(examplar)), dtype=np.float32)
    for i in tqdm(range(context["PATCHES_IN_TOTAL"])):
        mffcs[i, :] = librosa.feature.mfcc(y=patches[i], sr=SR).flatten()

    normalised = (mffcs.T / np.linalg.norm(mffcs, axis=1).T).T
    similarity = np.dot(normalised, normalised.T)
    np.fill_diagonal(similarity, 0.0)

    with open(f"batch/similarity_{context['name']}.npy", "wb") as f:
        np.save(f, similarity)

def compose(context, seeds):
    similarity = np.load(open(f"batch/similarity_{context['name']}.npy", "rb"))

    assert context["PATCHES_IN_COMPOSITION"] % len(seeds) == 0
    patches_per_seed = int(context["PATCHES_IN_COMPOSITION"] / len(seeds))

    composition = [seeds[0]]

    for seed_idx in range(len(seeds)):
        for _ in range(patches_per_seed):
            last_patch = composition[-1]
            next_patch = np.argmax(similarity[last_patch,:] + similarity[seeds[seed_idx],:])

            similarity[last_patch, next_patch] = 0.0
            similarity[next_patch, last_patch] = 0.0

            composition.append(int(next_patch))

    assert len(composition) == (context["PATCHES_IN_COMPOSITION"] + 1)
    json.dump(composition, open(f"out/composition_{context['name']}_seeds_{'_'.join([str(s) for s in seeds])}.json", "w"))

    mashed = []
    for i in range(1, len(composition), 2):
        j = i - 1

        sr, data1 = wavfile.read(f'batch/audio_{context["name"]}/{composition[j]}.wav')
        sr, data2 = wavfile.read(f'batch/audio_{context["name"]}/{composition[i]}.wav')

        if context["name"]=="flat":
            f = lambda data: fade_in_and_out(data[:int(0.15*sr)], sr, 0.001)
            data1 = f(data1); data2 = f(data2)

        if context["name"]=="bjork" or context["name"]=="jonnhy":
            f = lambda data: fade_in_and_out(data, sr, 0.001)
            data1 = f(data1); data2 = f(data2)

        elif context["name"]=="wave":
            f = lambda data: fade_out(fade_in(data, sr, 0.1), sr, 2.0)
            data1 = f(data1); data2 = f(data2)

        elif context["name"]=="point":
            f = lambda data: np.concatenate([data, np.zeros(int(sr*0.5))])
            data1 = f(data1); data2 = f(data2)

        if context["name"]=="flat":
            mashed.append(maad_crossfader(data1, data2, sr, 0.05))
        else:
            mashed.append(maad_crossfader(data1, data2, sr, 0.1))

    with open(f"out/audio_{context['name']}_seed_{'_'.join([str(s) for s in seeds])}.wav", 'wb') as f:
        f.write(Audio(np.concatenate(mashed), rate=sr).data)

def linear_crossfader(one, two):
    assert len(one) == len(two)
    assert len(one) % 2 == 0

    n_samples = len(one)
    n_transition_samples = n_samples/2
    n_constant_samples = (n_samples-n_transition_samples)/2

    fade_out_mask = np.concatenate([
        np.ones(ceil(n_constant_samples)),
        np.arange(1.0, 0.0, -1.0/n_transition_samples),
        np.zeros(floor(n_constant_samples)),
    ])
    fade_in_mask = 1.0 - fade_out_mask

    fade_out = one * fade_out_mask
    fade_in = two * fade_in_mask

    return fade_in + fade_out

def maad_crossfader(s1, s2, fs, fade_len):
    fade_in = np.sqrt(np.arange(0, fs * fade_len) / (fs * fade_len))
    fade_out = np.flip(fade_in)
    fstart_idx = int(len(s1) - (fs * fade_len) - 1)
    fstop_idx = int(fs * fade_len)
    s1_fade_out = s1[fstart_idx:-1] * fade_out
    s2_fade_in = s2[0:fstop_idx] * fade_in
    s_out = np.concatenate(
        [s1[0 : fstart_idx + 1], s1_fade_out + s2_fade_in, s2[fstop_idx - 1 : -1]]
    )
    return s_out

def fade_in(audio, sr, seconds_fading=8):
    samples_fading = int(sr*seconds_fading)
    fader = np.concatenate([
        np.arange(0.0, 1.0, 1.0/samples_fading),
        np.ones(len(audio)-samples_fading),
    ])
    return audio*fader

def fade_out(audio, sr, seconds_fading=8):
    samples_fading = int(sr*seconds_fading)
    fader = np.concatenate([
        np.ones(len(audio)-samples_fading),
        np.arange(1.0, 0.0, -1.0/samples_fading),
    ])
    return audio*fader

def fade_in_and_out(audio, sr, seconds_fading=8):
    return fade_out(fade_in(audio, sr, seconds_fading=seconds_fading), sr, seconds_fading=seconds_fading)

def mix_compositions(seeds):
    mashed = []
    for i in range(1, len(seeds)):
        j = i - 1

        sr, data1 = wavfile.read(f'out/audio_seed_{seeds[j]}.wav')
        sr, data2 = wavfile.read(f'out/audio_seed_{seeds[i]}.wav')

        mashed.append(
            linear_crossfader(data1, data2)
        )

    mix = np.concatenate(mashed)
    with_fading = fade_out(fade_in(mix, sr), sr)

    with open(f"out/audio_seeds_{'_'.join([str(e) for e in seeds])}.wav", 'wb') as f:
        f.write(Audio(with_fading, rate=sr).data)

if __name__=="__main__":
    # sample_space(point)
    # compute_similarity(flat)
    compose(flat, seeds=[89, 4, 20, 27, 42, 44, 54, 65, 115, 89])
    # compose(flat, seed=4)
    # compose(flat, seed=24)
    # compose(flat, seed=56)
    # compose(point, seed=0)

    # seeds = [4, 12, 20, 1]
    # mix_compositions(seeds)
    
    # seed = randint(0, PATCHES_IN_TOTAL-1)
    # seed = 20
    # compose(seed)
