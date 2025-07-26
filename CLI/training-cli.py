import subprocess
import json
import time
import os
import yaml
import getpass
from threading import Thread
from itertools import cycle
import random

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
    """Prints text in a large, white, ANSI Shadow font with purple gradient."""
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
    purple_colors = gradient.purple_gradient()
    for i, line in enumerate(lines):
        color = purple_colors[i % len(purple_colors)]
        print(color + line + style.END)
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
# SECTION 2: SIMPLE TRAINING PROGRESS INTERFACE
# ==============================================================================

class TrainingProgressMonitor:
    """Displays a simple progress bar from 0% to 100%."""
    
    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        
    def create_progress_bar(self, current, total, width=50):
        """Creates a visual progress bar."""
        progress = current / total if total > 0 else 0
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        percentage = progress * 100
        return f"[{Colors.COLOR_46}{bar}{Colors.END}] {percentage:.1f}%"
    
    def run_fake_training(self, config):
        """Runs the simple training progress simulation."""
        print(f"\n{Colors.COLOR_135}Starting training...{Colors.END}")
        print(f"Model: {config['base_model_id']}")
        print()
        
        try:
            for step in range(self.total_steps + 1):
                self.current_step = step
                progress_bar = self.create_progress_bar(step, self.total_steps, width=70)
                
                # Clear the line and print progress
                print(f"\rTraining Progress: {progress_bar} ({step}/{self.total_steps})", end="", flush=True)
                
                if step < self.total_steps:
                    time.sleep(0.1)  # Small delay for visual effect
            
            # Training complete
            print(f"\n\n{Colors.COLOR_46}🎉 Training completed successfully!{Colors.END}")
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.COLOR_196}Training stopped by user.{Colors.END}")

# ==============================================================================
# SECTION 3: VAST.AI MANAGER LOGIC
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
# SECTION 4: MODEL SELECTION
# ==============================================================================

def select_model():
    """Shows model selection menu and returns the selected model ID."""
    models = {
        '1': ('Mistral 7B Instruct', 'mistralai/Mistral-7B-Instruct-v0.2'),
        '2': ('LLAMA 8B', 'meta-llama/Llama-3.1-8B-Instruct'),
        '3': ('FLUX-1 Dev', 'black-forest-labs/FLUX.1-dev'),
        '4': ('Flux-1 Kontext', 'flux-1-kontext/flux-1-kontext'),
        '5': ('Kimi K2 Instruct', 'moonshot-ai/kimi-k2-instruct'),
        '6': ('Deepseek R1', 'deepseek-ai/deepseek-r1')
    }
    
    print_header("Model Selection")
    print(f"  {Colors.COLOR_135}Available Models:{Colors.END}")
    print()
    
    for key, (name, model_id) in models.items():
        print(f"  {Colors.COLOR_220}{key}.{Colors.END} {Colors.BRIGHT_WHITE}{name}{Colors.END}")
        print(f"     {Colors.COLOR_244}{model_id}{Colors.END}")
        print()
    
    while True:
        choice = input(f"  {style.WHITE}{style.BOLD}Select a model (1-6) [{style.END}{style.WHITE}1{style.BOLD}]:{style.END} ") or "1"
        
        if choice in models:
            selected_name, selected_id = models[choice]
            print(f"  {Colors.COLOR_46}✓{Colors.END} Selected: {Colors.BRIGHT_WHITE}{selected_name}{Colors.END}")
            return selected_id
        else:
            print_error(f"  Invalid choice. Please select 1-6.")

# ==============================================================================
# SECTION 5: MAIN CLI WORKFLOW
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
    
    # Use model selection function
    config['base_model_id'] = select_model()
    
    config['dataset_id'] = input(f"  {style.WHITE}{style.BOLD}Dataset ID [{style.END}{style.WHITE}databricks/databricks-dolly-15k{style.BOLD}]:{style.END} ") or "databricks/databricks-dolly-15k"
    config['lora_model_repo'] = input(f"  {style.WHITE}{style.BOLD}Output Repo Name [{style.END}{style.WHITE}your-hf-username/my-finetuned-model{style.BOLD}]:{style.END} ") or "your-hf-username/my-finetuned-model"
    config['gpu_name'] = input(f"  {style.WHITE}{style.BOLD}GPU Name [{style.END}{style.WHITE}RTX 4090{style.BOLD}]:{style.END} ") or "RTX 4090"
    config['num_gpus'] = int(input(f"  {style.WHITE}{style.BOLD}Number of GPUs [{style.END}{style.WHITE}1{style.BOLD}]:{style.END} ") or 1)
    
    return config

def main():
    """The main entry point for the Asimov CLI."""
    os.system('clear') # Clear the screen for a clean start
    
    # Show the ANSI title first
    print_ansi_shadow("ASIMOV")
    
    # Show model selection
    selected_model = select_model()
    
    # Default configuration for training interface
    config = {
        'base_model_id': selected_model,
        'dataset_id': 'databricks/databricks-dolly-15k',
        'lora_model_repo': 'your-hf-username/my-finetuned-model',
        'gpu_name': 'RTX 4090',
        'num_gpus': 1
    }
    
    # Launch the simple training progress interface
    monitor = TrainingProgressMonitor(total_steps=100)
    monitor.run_fake_training(config)

if __name__ == "__main__":
    main()
4