import os
import logging
from typing import Any, Literal
import json
import yaml

logger = logging.getLogger(__name__)


class FileManager():

    @staticmethod
    def read_yaml(filename: str) -> Any:
        """The `read_yaml` function reads a YAML file and returns its contents as a Python object, while handling any exceptions that may occur."""
        try:
            with open(filename, 'r', encoding='utf-8') as stream:
                data = yaml.safe_load(stream)
            return data
        except Exception as e:
            logger.error(f'Error in read_yaml: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
            return 'Error occured in read_yaml'

    @staticmethod
    def remove_files_from_folder(folder, file_extension) -> None:
        """Removes files with a specific file extension from a given folder.
        Args:folder (str): The path to the folder from which files will be removed.
            file_extension (str or Tuple[str]): The file extension(s) of the files to be removed."""
        try:
            full_folder_path = os.path.abspath(folder)
            logger.info(f"Full path of folder: {full_folder_path}")
            files_to_remove = []
            logger.info(f"Removing files from {folder} with extension: {file_extension}")
            for datafiles in os.listdir(folder):
                if datafiles.endswith(file_extension):
                    filepath = os.path.join(folder, datafiles)
                    if os.path.exists(filepath):
                        logger.info(f"File found: {datafiles} in {folder}")
                        files_to_remove.append(datafiles)
                        logger.info(f"File removed: {datafiles}")
                        input('Press Enter to continue...')
                    else:
                        logger.warning(f"File not found: {datafiles}")
            for datafiles in files_to_remove:
                filepath = os.path.join(folder, datafiles)
                os.remove(filepath)
            logger.info(f"Removed {len(files_to_remove)} files from {folder}")
        except Exception as e:
            logger.error(f'Error in remove_files_from_folder: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def save_json(data: Any, filename: str) -> None:
        """The `save_json` function saves data as a JSON file with error handling."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except IOError as e:
            logger.error(f"Error saving JSON file: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def read_json(filename: str) -> Any | Literal['Failed to read JSON file']:
        """The `read_json` function reads a JSON file and returns its contents, or a failure message if the
        file cannot be read."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except IOError as e:
            logger.error(f"Error reading JSON file: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
            return 'Failed to read JSON file'
