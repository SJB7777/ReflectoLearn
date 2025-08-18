"""
This module provides functionality to load and save configuration files,
with support for placeholder substitution in the configuration values.
"""
import re
import yaml
from reflectolearn.config.definitions import ExpConfig


class ConfigManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ConfigManager._initialized:
            self._config_file = None
            self._cached_config = None
            ConfigManager._initialized = True
    
    def initialize(self, config_file: str) -> 'ConfigManager':
        """
        Initialize the config manager with a configuration file path.
        
        Args:
            config_file (str): Path to the configuration file.
            
        Returns:
            ConfigManager: Self for method chaining.
        """
        self._config_file = config_file
        self._cached_config = None  # Clear cache when changing file
        return self
    
    def load_config(self, reload: bool = False) -> ExpConfig:
        """
        Load the configuration file with an option to reload the cache.

        Args:
            reload (bool): If True, reload the configuration and update the cache.

        Returns:
            ExpConfig: Configuration object with placeholders resolved.
            
        Raises:
            RuntimeError: If config file path is not set.
        """
        if self._config_file is None:
            raise RuntimeError("Config file not initialized. Call initialize() first.")
        
        if reload or self._cached_config is None:
            self._cached_config = self._load_config()
        
        return self._cached_config
    
    def _load_config(self) -> ExpConfig:
        """
        Load the configuration file, replace placeholders, and return the configuration object.

        Returns:
            ExpConfig: Configuration object with placeholders resolved.
        """
        with open(self._config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        context = config_dict.copy()
        replaced_data = self.replace_placeholders(config_dict, context)

        return ExpConfig(**replaced_data)
    
    def save_config(self, config_dict: dict) -> None:
        """
        Save the given configuration dictionary to the configuration file.

        Args:
            config_dict (dict): Configuration dictionary to be saved.
            
        Raises:
            RuntimeError: If config file path is not set.
        """
        if self._config_file is None:
            raise RuntimeError("Config file not initialized. Call initialize() first.")
        
        with open(self._config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        # Clear cache after saving
        self._cached_config = None
    
    def reload(self) -> ExpConfig:
        """
        Convenience method to reload the configuration.
        
        Returns:
            ExpConfig: Reloaded configuration object.
        """
        return self.load_config(reload=True)
    
    @staticmethod
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
                    lambda m: str(ConfigManager.resolve_placeholder(m.group(1), ctx)),
                    s,
                )
                if new_s == s:  # No changes, we're done
                    break
                s = new_s
            return s

        if isinstance(data, dict):
            # Create a copy of the data to avoid modifying the input
            result = {
                key: ConfigManager.replace_placeholders(value, context, max_iterations)
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
            return [ConfigManager.replace_placeholders(item, context, max_iterations) for item in data]
        elif isinstance(data, str):
            match = re.fullmatch(r"\$\{([^}]+)\}", data)
            if match:
                return ConfigManager.resolve_placeholder(match.group(1), context)
            return _replace_in_string(data, context)
        elif data is None or isinstance(data, (int, float, bool)):
            return data
        else:
            return data

    @staticmethod
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


# 편의 함수들 (기존 API 호환성 유지)
def initialize_config(config_file: str) -> ConfigManager:
    """
    Initialize the global config manager.
    
    Args:
        config_file (str): Path to the configuration file.
        
    Returns:
        ConfigManager: Initialized config manager.
    """
    return ConfigManager().initialize(config_file)


def load_config(reload: bool = False) -> ExpConfig:
    """
    Load the configuration using the global config manager.
    
    Args:
        reload (bool): If True, reload the configuration.
        
    Returns:
        ExpConfig: Configuration object.
    """
    return ConfigManager().load_config(reload=reload)


def save_config(config_dict: dict) -> None:
    """
    Save configuration using the global config manager.
    
    Args:
        config_dict (dict): Configuration dictionary to save.
    """
    ConfigManager().save_config(config_dict)


def reload_config() -> ExpConfig:
    """
    Reload the configuration using the global config manager.
    
    Returns:
        ExpConfig: Reloaded configuration object.
    """
    return ConfigManager().reload()


if __name__ == "__main__":
    # 사용 예시
    config_manager = initialize_config("config.yaml")
    config = load_config()
    print(config)
