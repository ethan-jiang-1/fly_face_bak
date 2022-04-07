import os


class MusicAmpDumper(object):
    @classmethod
    def dump_audio_info(cls, wav_filename, fps=24):
        import numpy as np
        from scipy.io import wavfile
        audio = {}

        if not os.path.exists(wav_filename):
            import moviepy.editor
            audio_clip = moviepy.editor.AudioFileClip(wav_filename)
            audio_clip.write_audiofile(wav_filename, fps=44100, nbytes=2, codec='pcm_s16le')

        track_name = os.path.basename(wav_filename)[:-4]
        rate, signal = wavfile.read(wav_filename)
        signal = np.mean(signal, axis=1) # to mono
        signal = np.abs(signal)
        # seed = signal.shape[0]
        duration = signal.shape[0] / rate
        frames = int(np.ceil(duration * fps))
        samples_per_frame = signal.shape[0] / frames
        audio[track_name] = np.zeros(frames, dtype=signal.dtype)
        for frame in range(frames):
            start = int(round(frame * samples_per_frame))
            stop = int(round((frame + 1) * samples_per_frame))
            audio[track_name][frame] = np.mean(signal[start:stop], axis=0)
        audio[track_name] /= max(audio[track_name])
        audio[track_name] = list(audio[track_name])

        ret = {}
        ret["audio"] = audio 
        ret["frames"] = frames 
        ret["trak_name"] = track_name

        return ret

def dump_music_to_json():
    import json
    names = [os.getcwd(), "exp_render_green_plant","mw_key2.wav"]

    wav_filename = os.sep.join(names)
    if not os.path.isfile(wav_filename):
        raise ValueError("filename exist {}".format(wav_filename))

    audio_info = MusicAmpDumper.dump_audio_info(wav_filename)
    json_filename = wav_filename + ".json"
    with open(json_filename, "wt+") as f:
        json.dump(audio_info, f, indent=4)

if __name__ == '__main__':
    dump_music_to_json()
