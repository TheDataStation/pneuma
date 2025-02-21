import os
import platform


def get_storage_path():
    if platform.system() == "Windows":
        return os.path.expanduser("~/Documents/Pneuma/out")
    else:  # Assuming Linux/Unix-like system
        xdg_data_home = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        return os.path.join(xdg_data_home, "Pneuma/out")
