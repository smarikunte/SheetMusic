from playsound import playsound
import numpy as np
import winsound

sounds = {
    (0, 0): 'sounds/melody.wav',
    (0, 1): 'sounds/high.wav',
    (1, 0): 'sounds/bassline.wav',
    (1, 1): 'sounds/lick.wav'
}


class AudioPlayer:
    def __init__(self):
        self.playing_x = None
        self.playing_y = None
        self.playing = np.zeros((2, 2))
        self.sounds = sounds

    def play_sound(self, pos, page_width, page_height):
        x_inc = page_width // 2
        y_inc = page_height // 2
        x = int(pos[0]) // x_inc
        y = int(pos[1]) // y_inc
        if x != self.playing_x:
            self.playing_x = x
        if y != self.playing_y:
            self.playing_y = y
        if 1 >= y >= 0 and 1 >= x >= 0 and not self.playing[y][x]:
            self.playing = np.zeros((2, 2))
            self.playing[y][x] = 1
            # winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(sounds[(y, x)], winsound.SND_ASYNC)

