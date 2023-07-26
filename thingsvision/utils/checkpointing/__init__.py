import os

ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def get_torch_home():
    """
    Gets the torch home folder used as a cache directory for model checkpoints.
    """
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(
                os.getenv(
                    ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR
                ),
                "torch",
            ),
        )
    )
    return torch_home
