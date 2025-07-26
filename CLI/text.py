import subprocess
import json
import time
import os
import yaml
import getpass
from threading import Thread
from itertools import cycle
from rich.console import Console
from rich.text import Text
from rich.color import Color

# ANSI escape codes for colors and styles
class Colors:
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'
    
    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    RESET = '\033[0m'
    END = '\033[0m'
    
    # 256-color palette (examples from the image)
    # Reds
    COLOR_196 = '\033[38;5;196m'  # Bright red
    COLOR_197 = '\033[38;5;197m'
    COLOR_198 = '\033[38;5;198m'
    COLOR_199 = '\033[38;5;199m'
    
    # Greens
    COLOR_46 = '\033[38;5;46m'   # Bright green
    COLOR_47 = '\033[38;5;47m'
    COLOR_82 = '\033[38;5;82m'
    COLOR_83 = '\033[38;5;83m'
    
    # Blues
    COLOR_21 = '\033[38;5;21m'   # Bright blue
    COLOR_27 = '\033[38;5;27m'
    COLOR_33 = '\033[38;5;33m'
    COLOR_39 = '\033[38;5;39m'
    
    # Yellows/Orange
    COLOR_220 = '\033[38;5;220m'  # Yellow
    COLOR_214 = '\033[38;5;214m'  # Orange
    COLOR_208 = '\033[38;5;208m'
    COLOR_202 = '\033[38;5;202m'
    
    # Purples/Magentas
    COLOR_129 = '\033[38;5;129m'
    COLOR_135 = '\033[38;5;135m'
    COLOR_141 = '\033[38;5;141m'
    COLOR_147 = '\033[38;5;147m'
    
    # Cyans
    COLOR_51 = '\033[38;5;51m'
    COLOR_87 = '\033[38;5;87m'
    COLOR_123 = '\033[38;5;123m'
    COLOR_159 = '\033[38;5;159m'
    
    # Grays
    COLOR_232 = '\033[38;5;232m'  # Very dark gray
    COLOR_238 = '\033[38;5;238m'  # Dark gray
    COLOR_244 = '\033[38;5;244m'  # Medium gray
    COLOR_250 = '\033[38;5;250m'  # Light gray
    COLOR_255 = '\033[38;5;255m'  # Very light gray
    
    @staticmethod
    def color_256(color_code):
        """Return ANSI escape sequence for any 256-color code (0-255)"""
        return f'\033[38;5;{color_code}m'
    
    @staticmethod
    def bg_color_256(color_code):
        """Return ANSI escape sequence for any 256-color background (0-255)"""
        return f'\033[48;5;{color_code}m'
    
    @staticmethod
    def rgb(r, g, b):
        """Return ANSI escape sequence for RGB color (0-255 for each component)"""
        return f'\033[38;2;{r};{g};{b}m'
    
    @staticmethod
    def bg_rgb(r, g, b):
        """Return ANSI escape sequence for RGB background color (0-255 for each component)"""
        return f'\033[48;2;{r};{g};{b}m'

# Legacy compatibility aliases
class style:
    BLUE = Colors.BRIGHT_BLUE
    GREEN = Colors.BRIGHT_GREEN
    YELLOW = Colors.BRIGHT_YELLOW
    RED = Colors.BRIGHT_RED
    PURPLE = Colors.BRIGHT_MAGENTA
    WHITE = Colors.BRIGHT_WHITE
    BOLD = Colors.BOLD
    UNDERLINE = Colors.UNDERLINE
    END = Colors.END

class gradient:
    @staticmethod
    def purple_gradient():
        """Returns a purple gradient using available purple colors"""
        return [Colors.COLOR_129, Colors.COLOR_135, Colors.COLOR_141, Colors.COLOR_147]
    
    def __init__(self, start_color, end_color):
        start_color = Colors.COLOR_129
        self.end_color = Colors.COLOR_135
        self.current_color = start_color
        self.step = 0.01
        self.max_steps = 100
        self.current_step = 0
        self.text = ""
        self.text_length = 0
        self.text_index = 0
        self.text_speed = 0.01
        self.text_delay = 0.01

# ==============================================================================
# SECTION 1: STYLING AND UI HELPERS
# ==============================================================================

def print_ansi_shadow(text):
    """Prints text with smooth purple gradient using rich library."""
    console = Console()
    
    chars = {
        'A': [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
        'S': ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
        'I': ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
        'M': ["███╗   ███╗", "████╗ ████║", "██╔████╔██║", "██║╚██╔╝██║", "██║ ╚═╝ ██║", "╚═╝     ╚═╝"],
        'O': [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
        'V': ["██╗   ██╗", "██║   ██║", "██║   ██║", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚═══╝  "],
        'C': [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
        'L': ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
        ' ': ["   ", "   ", "   ", "   ", "   ", "   "]
    }
    
    lines = ["", "", "", "", "", ""]
    for char in text.upper():
        if char in chars:
            for i, line in enumerate(chars[char]):
                lines[i] += line + " "
    
    print("\n")
    
    # Purple gradient colors (dark purple to light purple)
    purple_shades = [
        "#9185ff",  # Indigo
        "#5338ff",  # Lavender
    ]
    
    for line_index, line in enumerate(lines):
        # Calculate which color this entire line should be
        vertical_position = line_index / max(1, len(lines) - 1)
        color_position = vertical_position * (len(purple_shades) - 1)
        
        # Get interpolated color for this line
        if color_position <= len(purple_shades) - 1:
            lower_index = int(color_position)
            
            if lower_index < len(purple_shades) - 1:
                # Interpolate between two colors
                upper_index = lower_index + 1
                current_color = Color.parse(purple_shades[lower_index])
                next_color = Color.parse(purple_shades[upper_index])
                
                # Calculate interpolation factor
                interpolation = color_position - lower_index
                
                # Interpolate RGB values
                r = int(current_color.triplet.red + (next_color.triplet.red - current_color.triplet.red) * interpolation)
                g = int(current_color.triplet.green + (next_color.triplet.green - current_color.triplet.green) * interpolation)
                b = int(current_color.triplet.blue + (next_color.triplet.blue - current_color.triplet.blue) * interpolation)
                
                line_color = f"rgb({r},{g},{b})"
            else:
                line_color = purple_shades[-1]
        else:
            line_color = purple_shades[-1]
        
        # Apply the SAME color to every character in this line
        gradient_text = Text()
        for char in line:
            gradient_text.append(char, style=line_color)
        
        console.print(gradient_text)
    
    print("\n")

class Spinner:
    """A simple spinner class to show progress for long-running tasks."""
    def __init__(self, message="Loading..."):
        self.message = message
        self.steps = cycle(['-', '\\', '|', '/'])
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = Thread(target=self._spin)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        print("\r" + " " * (len(self.message) + 2), end="\r")

    def _spin(self):
        while self.running:
            print(f"\r{style.WHITE}{next(self.steps)}{style.END} {style.WHITE}{self.message}{style.END}", end="")
            time.sleep(0.1)

def print_success(message):
    print(f"{style.WHITE}✓{style.END} {style.WHITE}{message}{style.END}")

def print_error(message):
    print(f"{style.WHITE}✗{style.END} {style.WHITE}{message}{style.END}")

def print_info(message):
    print(f"{style.WHITE}ℹ{style.END} {style.WHITE}{message}{style.END}")

def print_header(title):
    print(f"\n{style.WHITE}{style.BOLD}{style.UNDERLINE}{title}{style.END}")

# ==============================================================================
# SECTION 2: VAST.AI MANAGER LOGIC
# (This is a simplified version of our vast_manager.py)
# ==============================================================================

def search_cheapest_instance(gpu_name, num_gpus):
    """Searches for the cheapest available interruptible instance on Vast.ai."""
    query = f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true'
    command = ["vastai", "search", "offers", query, "--interruptible", "--order", "dph_total asc", "--raw"]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            return None, None
        instances = json.loads(result.stdout)
        if not instances:
            return None, None
        cheapest = instances[0]
        return cheapest['id'], cheapest['dph_total']
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None, None

def create_instance(instance_id, docker_image, env_vars, bid_price):
    """Creates a Vast.ai instance by placing a bid."""
    export_cmds = "; ".join([f"export {key}='{value}'" for key, value in env_vars.items()])
    training_cmd = "python /app/train.py"
    final_on_start_cmd = f"{export_cmds}; {training_cmd}"

    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "50",
        "--bid", str(bid_price),
        "--onstart-cmd", final_on_start_cmd,
        "--raw"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        instance_info = json.loads(result.stdout)
        if instance_info.get("success"):
            return instance_info['new_contract']
        else:
            return None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None

# ==============================================================================
# SECTION 3: MAIN CLI WORKFLOW
# ==============================================================================

def get_user_config():
    """Interactively prompts the user for all necessary job configuration."""
    config = {}
    print_header("Job Configuration")
    
    # Use getpass for the API key so it's not visible on screen
    while True:
        try:
            hf_token = getpass.getpass(f"  {style.WHITE}{style.BOLD}Enter your {style.YELLOW}Hugging Face{style.WHITE} Token to import Models and Datasets (hf_...):{style.END} ")
            if hf_token.startswith("hf_") and len(hf_token) > 10:
                config['hf_token'] = hf_token
                break
            else:
                print_error("  Invalid token format. It must start with 'hf_'. Please try again.")
        except (EOFError, KeyboardInterrupt):
            return None

    config['docker_image'] = input(f"  {style.WHITE}{style.BOLD}Docker Image [{style.END}{style.WHITE}halez/lora-finetuner:latest{style.BOLD}]:{style.END} ") or "halez/lora-finetuner:latest"
    config['base_model_id'] = input(f"  {style.WHITE}{style.BOLD}Base Model ID [{style.END}{style.WHITE}mistralai/Mistral-7B-Instruct-v0.2{style.BOLD}]:{style.END} ") or "mistralai/Mistral-7B-Instruct-v0.2"
    config['dataset_id'] = input(f"  {style.WHITE}{style.BOLD}Dataset ID [{style.END}{style.WHITE}databricks/databricks-dolly-15k{style.BOLD}]:{style.END} ") or "databricks/databricks-dolly-15k"
    config['lora_model_repo'] = input(f"  {style.WHITE}{style.BOLD}Output Repo Name [{style.END}{style.WHITE}your-hf-username/my-finetuned-model{style.BOLD}]:{style.END} ") or "your-hf-username/my-finetuned-model"
    config['gpu_name'] = input(f"  {style.WHITE}{style.BOLD}GPU Name [{style.END}{style.WHITE}RTX 4090{style.BOLD}]:{style.END} ") or "RTX 4090"
    config['num_gpus'] = int(input(f"  {style.WHITE}{style.BOLD}Number of GPUs [{style.END}{style.WHITE}1{style.BOLD}]:{style.END} ") or 1)
    
    return config

def main():
    """The main entry point for the Asimov CLI."""
    os.system('clear') # Clear the screen for a clean start
    print_ansi_shadow("ASIMOV CLI")
    print_info("Welcome to the Asimov Fine-tuning CLI.")
    print(f"{Colors.COLOR_141}This tool will help you to train open-source models in just few clicks.{style.END}")

    try:
        config = get_user_config()
        if config is None:
            print_error("\nConfiguration cancelled. Exiting.")
            return

        print_header("Starting Job")
        
        spinner = Spinner("Searching for the cheapest available GPU...")
        spinner.start()
        instance_id, bid_price = search_cheapest_instance(config['gpu_name'], config['num_gpus'])
        spinner.stop()

        if not instance_id:
            print_error("Could not find a suitable GPU. Please try a different GPU type or try again later.")
            return
        
        print_success(f"Found interruptible instance {instance_id} at ${bid_price:.4f}/hr")

        env_vars = {
            "HF_TOKEN": config['hf_token'],
            "BASE_MODEL_ID": config['base_model_id'],
            "DATASET_ID": config['dataset_id'],
            "LORA_MODEL_REPO": config['lora_model_repo']
        }

        spinner = Spinner(f"Creating instance {instance_id}...")
        spinner.start()
        new_instance_id = create_instance(instance_id, config['docker_image'], env_vars, bid_price)
        spinner.stop()

        if not new_instance_id:
            print_error("Failed to create the instance on Vast.ai.")
            return

        print_success(f"Instance {new_instance_id} created successfully!")
        print_header("Next Steps")
        print(f"{style.WHITE}Your fine-tuning job is now running in the background on Vast.ai.{style.END}")
        print(f"{style.WHITE}You can monitor its progress with the following command:{style.END}")
        print(f"\n  {style.WHITE}{style.BOLD}vastai logs -f {new_instance_id}{style.END}\n")

    except (KeyboardInterrupt, EOFError):
        print_error("\n\nOperation cancelled by user. Exiting.")
    except Exception as e:
        print_error(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
