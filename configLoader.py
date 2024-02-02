from configurator import excelConfigurator, regexConfigurator, load_config_from_file
import os
import logging
import stat

logger = logging.getLogger(__name__)
CONFIG_FILE = 'config/config.yaml'


class ConfigLoader():

    @staticmethod
    def load_configurations():
        """The function "load_configurations" returns three objects: an excelConfigurator object, a
        regexConfigurator object, and the result of loading a configuration from a file."""
        config_keys = ['CL_INIT_ORDER', 'CL_DROPPED', 'CL_FINAL_ORDER', 'EXCLUSION_PAIRS']
        regex_keys = ['MATCH_VALUE_TO_SPECIFIC_COLUMN']
        return excelConfigurator(config_keys), regexConfigurator(regex_keys), load_config_from_file(CONFIG_FILE)


class SystemUtils():

    @staticmethod
    def check_and_modify_permissions(path) -> None:
        """Check and modify the permissions of the given path.
        Args: path: The path to check and modify permissions for."""
        try:
            if not os.access(path, os.W_OK):
                logger.info(f"Write permission not enabled on {path}. Attempting to modify.")
                current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
                os.chmod(path, current_permissions | stat.S_IWUSR)
                logger.info(f"Write permission added to {path}.")
            else:
                logger.info(f"Write permission is enabled on {path}.")
        except Exception as e:
            logger.info(f"Error modifying permissions: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
            logger.error(f"Error modifying permissions for {path}: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
