import argparse
import os
import sys
import pyfiglet

# Popular pyfiglet font styles
FONTS = {
    '1': 'standard',
    '2': 'slant',
    '3': 'block',
    '4': 'bubble',
    '5': 'digital',
    '6': 'shadow',
    '7': '3d-diagonal',
    '8': 'script'
}

# ANSI color codes
COLORS = {
    'cyan': '\033[96m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'magenta': '\033[95m',
    'blue': '\033[94m',
    'reset': '\033[0m'
}

def colorize(text: str, color_name: str) -> str:
    color_code = COLORS.get(color_name.lower(), '')
    if color_code:
        return f"{color_code}{text}{COLORS['reset']}"
    return text

def display_header():
    print('-' * 70)
    ascii_banner = pyfiglet.figlet_format("ASCII Banner")
    print(colorize(ascii_banner, 'cyan'))
    print('-' * 70)

def generate_banner(text: str, font: str = 'standard') -> str:
    try:
        return pyfiglet.figlet_format(text, font=font)
    except pyfiglet.FontNotFound:
        return pyfiglet.figlet_format(text, font='standard')

def save_to_file(banner_text: str):
    filename = input("Enter filename to save (e.g. banner.txt) or press Enter to skip: ").strip()
    if filename:
        if not filename.endswith('.txt'):
            filename += '.txt'
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(banner_text)
            print(colorize(f"Successfully saved ASCII art to {filename}", 'green'))
        except Exception as e:
            print(colorize(f"Failed to save file: {e}", 'red'))

def select_font() -> str:
    print("\nSelect Font Style:")
    for num, font_name in FONTS.items():
        print(f"  [{num}] {font_name}")
    choice = input("Enter font number (default 1): ").strip()
    return FONTS.get(choice, 'standard')

def select_color() -> str:
    print("\nSelect Color:")
    color_keys = ['cyan', 'green', 'yellow', 'red', 'magenta', 'blue']
    for idx, col in enumerate(color_keys, 1):
        print(f"  [{idx}] {col.capitalize()}")
    choice = input("Enter color number (default 1): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(color_keys):
            return color_keys[idx]
    except ValueError:
        pass
    return 'cyan'

def interactive_mode():
    display_header()
    while True:
        text = input("\nEnter Your Text: ").strip()
        if not text:
            print("Text cannot be empty!")
            continue

        font = select_font()
        color = select_color()

        raw_banner = generate_banner(text, font=font)
        colored_banner = colorize(raw_banner, color)

        print("\nGenerated Banner:\n")
        print(colored_banner)

        save_to_file(raw_banner)

        again = input("\nDo you want to generate another text? (y/n): ").strip().lower()
        if again != 'y':
            print("\nThanks for using Fancy Text Generator! :)\n")
            break

def main():
    parser = argparse.ArgumentParser(description="Fancy Text Generator - Transform text into ASCII banner art.")
    parser.add_argument('-t', '--text', type=str, help="Text to convert into ASCII art")
    parser.add_argument('-f', '--font', type=str, default='standard', help="Font style (standard, slant, block, bubble, digital, shadow, 3d-diagonal, script)")
    parser.add_argument('-c', '--color', type=str, default='cyan', help="Color (cyan, green, yellow, red, magenta, blue)")
    parser.add_argument('-o', '--output', type=str, help="Output file path to save ASCII art")

    args = parser.parse_args()

    if args.text:
        raw_banner = generate_banner(args.text, font=args.font)
        print(colorize(raw_banner, args.color))
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(raw_banner)
            print(f"Saved to {args.output}")
    else:
        interactive_mode()

if __name__ == '__main__':
    main()
