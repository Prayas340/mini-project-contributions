import os
import sys
import time as t
import click

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def play_audio(filename: str, extra_wait_factor: float = 1.0):
    if not AUDIO_AVAILABLE:
        return
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        return
    try:
        sound = pygame.mixer.Sound(file_path)
        sound.play()
        wait_ms = int(sound.get_length() * 1000 * extra_wait_factor)
        pygame.time.wait(wait_ms)
    except Exception:
        pass

def countdown(seconds: int, prefix: str = ""):
    for remaining in range(seconds, 0, -1):
        sys.stdout.write(f"\r{prefix}{remaining}s left...")
        sys.stdout.flush()
        t.sleep(1)
    sys.stdout.write("\r" + " " * (len(prefix) + 15) + "\r")
    sys.stdout.flush()

@click.command()
@click.option('--time', '-t', default=10, type=int, help='Time you want to exercise (seconds)')
@click.option('--interval', '-i', default=3, type=int, help='Interval rest time between reps (seconds)')
@click.option('--reps', '-r', default=5, type=int, help='Number of reps to perform')
def exercise(time: int, interval: int, reps: int):
    if time <= 0:
        click.echo("Error: --time must be greater than 0.")
        return
    if interval < 0:
        click.echo("Error: --interval must be non-negative.")
        return
    if reps <= 0:
        click.echo("Error: --reps must be greater than 0.")
        return

    total_reps = reps
    for current_rep in range(1, total_reps + 1):
        print(f"\n--- Rep {current_rep}/{total_reps} ---")
        print('Start!')
        play_audio("beep_start.mp3")
        countdown(time, prefix="Exercising: ")
        print('Stop!')
        play_audio("beep_stop.mp3")

        reps_left = total_reps - current_rep
        if reps_left > 0:
            print(f'Reps left: {reps_left}')
            if interval > 0:
                print('Resting...')
                countdown(interval, prefix="Resting: ")
        else:
            print("\nFinished!")
            play_audio("beep_finish.mp3", extra_wait_factor=1.5)

if __name__ == '__main__':
    exercise()
