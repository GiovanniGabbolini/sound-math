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

PATCHES_IN_TOTAL = 9000

def sample_space():
    server = Server()

    # Recording properties
    SAMPLE_RATE = 44100
    SECONDS = 0.5
    # Recording device
    REC.default.device = 'Soundflower (2ch)'

    i = 0
    while i < PATCHES_IN_TOTAL:
        patch = sample_params()
        synth = Synth(server, "superfm", patch)

        recording = REC.rec( int(SECONDS * SAMPLE_RATE), samplerate = SAMPLE_RATE, channels = 1)
        REC.wait()

        write(f"batch/audio/{i}.wav", SAMPLE_RATE, recording)
        json.dump(patch, open(f"batch/patches/{i}.json", "w"))

        synth.free()

        i += 1

def compose(seed):
    PATCHES_IN_COMPOSITION = 300
    SR = 44100

    patches = np.zeros((PATCHES_IN_TOTAL, 22050), dtype=np.float32)
    for i in tqdm(range(PATCHES_IN_TOTAL)):
        _, data = wavfile.read(f'batch/audio/{i}.wav') # load the data
        patches[i]=data
    patches = (patches.T - np.mean(patches, axis=1)).T # subtract mean

    mffcs = np.zeros((PATCHES_IN_TOTAL, 20*44), dtype=np.float32)
    for i in tqdm(range(PATCHES_IN_TOTAL)):
        mffcs[i, :] = librosa.feature.mfcc(y=patches[i], sr=SR).flatten()
    
    normalised = (mffcs.T / np.linalg.norm(mffcs, axis=1).T).T
    similarity = np.dot(normalised, normalised.T)
    np.fill_diagonal(similarity, 0.0)

    composition = [seed]

    while len(composition) < PATCHES_IN_COMPOSITION:
        last_patch = composition[-1]
        next_patch = np.argmax(similarity[last_patch,:] + similarity[seed,:])

        similarity[last_patch, next_patch] = 0.0
        similarity[next_patch, last_patch] = 0.0
        
        composition.append(int(next_patch))

    json.dump(composition, open(f"out/composition_seed_{seed}.json", "w"))

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

def fade_in(audio, sr):
    samples_fading = sr*8
    fader = np.concatenate([
        np.arange(0.0, 1.0, 1.0/samples_fading),
        np.ones(len(audio)-samples_fading),
    ])
    return audio*fader

def fade_out(audio, sr):
    samples_fading = sr*8
    fader = np.concatenate([
        np.ones(len(audio)-samples_fading),
        np.arange(1.0, 0.0, -1.0/samples_fading),
    ])
    return audio*fader

def record_composition(seed):
    composition = json.load(open(f"out/composition_seed_{seed}.json"))

    mashed = []
    for i in range(1, len(composition)):
        j = i - 1

        sr, data1 = wavfile.read(f'batch/audio/{composition[j]}.wav')
        sr, data2 = wavfile.read(f'batch/audio/{composition[i]}.wav')

        data1 = data1[:8000]
        data2 = data2[:8000]

        mashed.append(
            maad_crossfader(data1, data2, sr, 0.1)
        )

    with open(f"out/audio_seed_{seed}.wav", 'wb') as f:
        f.write(Audio(np.concatenate(mashed), rate=sr).data)

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
    seeds = [4, 12, 20, 1]
    mix_compositions(seeds)
    
    # seed = randint(0, PATCHES_IN_TOTAL-1)
    # seed = 20
    # compose(seed)
    # record_composition(seed)