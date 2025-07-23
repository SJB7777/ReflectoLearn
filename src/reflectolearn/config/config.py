"""
This module provides functionality to load and save configuration files,
with support for placeholder substitution in the configuration values.
"""

import re
from functools import lru_cache
import yaml
from reflectolearn.config.definitions import ExpConfig


def replace_placeholders(data, context, max_iterations=10):
    """
    Recursively replace placeholders in the given data structure.

    Placeholders are in the format `${key.subkey}`.

    Args:
        data (dict, list, str, or any): Data structure containing placeholders.
        context (dict): Context dictionary for placeholder resolution.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        Data structure with placeholders replaced.
    """

    def _replace_in_string(s, ctx):
        # Keep replacing placeholders until no more changes occur
        for _ in range(max_iterations):
            new_s = re.sub(
                r"\$\{([^}]+)\}",
                lambda m: str(resolve_placeholder(m.group(1), ctx)),
                s,
            )
            if new_s == s:  # No changes, we're done
                break
            s = new_s
        return s

    if isinstance(data, dict):
        # Create a copy of the data to avoid modifying the input
        result = {
            key: replace_placeholders(value, context, max_iterations)
            for key, value in data.items()
        }
        # Update context with resolved values for subsequent resolutions
        context.update(
            {
                k: v
                for k, v in result.items()
                if isinstance(v, (dict, str, int, float, bool))
            }
        )
        return result
    elif isinstance(data, list):
        return [replace_placeholders(item, context, max_iterations) for item in data]
    elif isinstance(data, str):
        match = re.fullmatch(r"\$\{([^}]+)\}", data)
        if match:
            return resolve_placeholder(match.group(1), context)
        return _replace_in_string(data, context)
    elif data is None or isinstance(data, (int, float, bool)):
        return data
    else:
        return data


def resolve_placeholder(placeholder, context):
    """
    Resolve a placeholder to its corresponding value from the context.

    Args:
        placeholder (str): Placeholder string in the format `${key.subkey}`.
        context (dict): Context dictionary containing the values.

    Returns:
        Value corresponding to the placeholder.

    Raises:
        ValueError: If the placeholder cannot be resolved.
    """
    keys = placeholder.split(".")
    value = context
    for key in keys:
        value = value.get(key)
        if value is None:
            raise ValueError(f"Placeholder '{placeholder}' could not be resolved")
    return value


@lru_cache(maxsize=1)
def __load_config() -> ExpConfig:
    """
    Load the configuration file, replace placeholders, and return the configuration object.

    Returns:
        ExpConfig: Configuration object with placeholders resolved.
    """
    config_file = "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    context = config_dict.copy()
    replaced_data = replace_placeholders(config_dict, context)

    return ExpConfig(**replaced_data)


def load_config(reload: bool = False) -> ExpConfig:
    """
    Load the configuration file with an option to reload the cache.

    Args:
        reload (bool): If True, reload the configuration and update the cache.

    Returns:
        ExpConfig: Configuration object with placeholders resolved.
    """
    if reload:
        __load_config.cache_clear()
        return __load_config()
    return __load_config()


def save_config(config_dict: dict) -> None:
    """
    Save the given configuration dictionary to the configuration file.

    Args:
        config_dict (dict): Configuration dictionary to be saved.
    """
    config_file = "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    config = load_config()
    print(config)
