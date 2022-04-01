
class MusicMp4Mixer(object):
    @classmethod
    def mix_music_to_mp4(cls, mp4_input_filename, wav_input_filename, mp4_output, fps=24):
        import moviepy.editor
        video_clip = moviepy.editor.VideoFileClip(mp4_input_filename)
        audio_clip_i = moviepy.editor.AudioFileClip(wav_input_filename)
        video_clip = video_clip.set_audio(audio_clip_i)
        video_clip.write_videofile(mp4_output, fps=fps, codec='libx264', audio_codec='aac', bitrate='15M')


def mix_wav_to_mp4():
    import os
    names = [os.getcwd(), "exp_render_green_plant","mw_key2.wav"]
    wav_input_filename = os.sep.join(names)

    names = [os.getcwd(), "_gen_render_mp4","output_gen_render_green_plant.mp4"]
    mp4_input_filename = os.sep.join(names)

    names = [os.getcwd(), "_gen_render_mp4","output_gen_render_green_plant_mixed.mp4"]
    mp4_output = os.sep.join(names)

    MusicMp4Mixer.mix_music_to_mp4(mp4_input_filename, wav_input_filename, mp4_output)

if __name__ == '__main__':
    mix_wav_to_mp4()
