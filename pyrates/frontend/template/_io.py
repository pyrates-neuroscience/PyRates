"""Some utility functions for template loading/writing"""


def _complete_template_path(target_path: str, source_path: str) -> str:
    """Check if path contains a folder structure and prepend own path, if it doesn't"""

    if "." not in target_path:
        if "/" in source_path or "\\" in source_path:
            import os
            basedir, _ = os.path.split(source_path)
            target_path = os.path.normpath(os.path.join(basedir, target_path))
        else:
            target_path = ".".join((*source_path.split('.')[:-1], target_path))
    return target_path