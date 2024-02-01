import logging
import shutil
import os
from datetime import datetime
from typing import Any, Literal, LiteralString
from pathlib import Path
import pandas as pd
from timeManager import TimeManager, TODAY
from configLoader import SystemUtils

logger = logging.getLogger(__name__)

LOG_FOLDER = 'AZLOGS'
is_date_string = TimeManager.is_date_string


class BackupManager():

    @staticmethod
    def get_date_folders(input_dir, date_format='%d-%m-%y') -> list[Any]:  # for backups
        """Identify and return folders that match the given date format in their name."""
        date_folders = []
        for root, dirs, _ in os.walk(input_dir):
            for dir_name in dirs:
                if is_date_string(dir_name, date_format):
                    folder_path = os.path.join(root, dir_name)
                    date_folders.append(folder_path)
        return date_folders

    @staticmethod
    def copy_log_file(source_file, destination_file) -> LiteralString | Literal['Error: Log file not found']:
        """The `copy_log_file` function copies a log file from a source file path to a destination file path
        and returns a success message or an error message if the log file is not found."""
        log_file = 'applogs.log'
        log_file_path = source_file
        backup_log_file_path = destination_file

        if os.path.exists(log_file_path):
            shutil.copy(log_file_path, backup_log_file_path)
            logger.info(f'Log file copied: {log_file}')
            return f'Log file copied: {log_file}'
        else:
            logger.warning(f'Log file not found: {log_file}')
            return 'Error: Log file not found'

    @staticmethod
    def create_backups(input_dir, backup_dir, query_name) -> None:
        """The `create_backups` function creates backups of specific file types in a given input directory,
        using a unique filename based on the current timestamp and other parameters."""
        SystemUtils.check_and_modify_permissions(backup_dir)

        files = os.listdir(input_dir)
        logger.info(f'input_dir {input_dir} - backup_dir {backup_dir}')
        logger.info(f'Files found: {files}')
        moved_files = []
        AZDIAG_CSV_FILEPATH_STR = 'AzDiag.csv'
        if AZDIAG_CSV_FILEPATH_STR in files:
            csv_file_path = os.path.join(input_dir, AZDIAG_CSV_FILEPATH_STR)
            df = pd.read_csv(csv_file_path, low_memory=False)
            # df = pd.read_csv(os.path.join(input_dir, AZDIAG_CSV_FILEPATH_STR))

            logger.info(f'First LocalTime found: {df["LocalTime"].iloc[0]}')
            df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%d-%m-%y %H:%M:%S', errors='coerce')
            first_timestamp = df['LocalTime'].iloc[0].strftime('%d-%m-%y_%I.%M%p')

            logger.info(f'File loaded: {AZDIAG_CSV_FILEPATH_STR} which has {df.shape[0]} rows and {df.shape[1]} columns')
            logger.info(f'Converted timestamp: {first_timestamp}')

            extensions_to_copy = ['.csv', '.json', '.xlsx', '.log', '.txt']  # ! EXTENSION LIST
            for file in files:
                filename, file_extension = os.path.splitext(file)
                if file_extension in extensions_to_copy:
                    # ! Check if "FINAL" should be included in the filename.
                    is_final = 'final' in filename.lower() or 'FINAL' in filename.lower()
                    final_suffix = "_FINAL" if is_final else ""
                    base_final_filename = f"{query_name}{final_suffix}_{first_timestamp}{file_extension}"
                    final_filename = base_final_filename

                    # ! Ensure the filename is unique within the backup directory.
                    c_to = os.path.join(backup_dir, final_filename)
                    counter = 1
                    while os.path.exists(c_to):
                        final_filename = f"{filename}_{counter}{final_suffix}_{first_timestamp}{file_extension}"
                        c_to = os.path.join(backup_dir, final_filename)
                        counter += 1

                    shutil.copy(os.path.join(input_dir, file), c_to)
                    logger.info(f'File copied: {final_filename}')
                    moved_files.append(final_filename)

        logger.info(f'Following files have been moved: {", ".join(moved_files)}')

    @staticmethod
    def store_old_backups(input_dir, days_threshold=7) -> None:
        """The `store_old_backups` function takes an input directory and a days threshold, and archives and
            deletes folders that are older than the threshold."""
        backup_folders = [folder for folder, _, _ in os.walk(input_dir) if os.path.isdir(folder) and folder != input_dir]
        logger.debug(f'backup_folders: {backup_folders}')
        date_format = '%d-%m-%y'
        date_folders = BackupManager.get_date_folders(input_dir, date_format)

        for folder in date_folders:
            try:
                folder_date = datetime.strptime(os.path.basename(folder), date_format)
                current_date = datetime.now()
                date_difference = current_date - folder_date
                logger.debug(f'Date difference for {folder}: {date_difference.days} days')

                if date_difference.days >= days_threshold:
                    shutil.make_archive(folder, 'zip', folder)
                    shutil.rmtree(folder)
                    logger.debug(f'Folder archived and deleted: {folder}')
                else:
                    logger.info(f'Folder within threshold, skipping: {folder}')
            except ValueError:
                logger.error(f'Folder name does not match date format and will be skipped: {folder}')

    @staticmethod
    def manage_backups(output_folder: str, query_name: str) -> None:
        """The `manage_backups` function creates backup directories, performs backups, stores old backups, and
        copies log files."""
        kql_backup_dir = f'{output_folder}/{query_name}'
        if Path(kql_backup_dir).is_dir():
            logger.info(f'Backup dir exists: {kql_backup_dir}')
        else:
            logger.info(f'Backup dir does not exist: {kql_backup_dir}'
                        f'\nCreating dir: {kql_backup_dir}')
            Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
        backup_dir = f'{kql_backup_dir}/{TODAY}'
        if Path(backup_dir).is_dir():
            logger.info(f'Backup dir exists: {backup_dir}')
        else:
            logger.info(f'Backup dir does not exist: {backup_dir}'
                        f'\nCreating dir: {backup_dir}')
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f'Backup dir: {backup_dir}')
        if not Path(backup_dir).is_dir() and not Path(kql_backup_dir).is_dir():  # ? NOTE
            Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
        # if not Path(backup_dir).is_dir():
        #     if not Path(kql_backup_dir).is_dir():
        #         Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
        #         Path(backup_dir).mkdir(parents=True, exist_ok=True)
        try:
            SRC_LOG_FILE = f'{LOG_FOLDER}/applogs.log'
            DEST_LOG_FILE = f'{backup_dir}/applogs.log'
            BackupManager.create_backups(input_dir=output_folder, backup_dir=backup_dir, query_name=query_name)
            logger.info(f'Backup completed: {backup_dir}')
            BackupManager.store_old_backups(input_dir=output_folder, days_threshold=7)
            logger.info(f'Old backups stored: {output_folder}')
            BackupManager.copy_log_file(source_file=SRC_LOG_FILE, destination_file=DEST_LOG_FILE)
            logger.info(f'copy_log_file: {SRC_LOG_FILE} to {DEST_LOG_FILE}')
        except Exception as err:
            logger.error(f'Error in manage_backups: {err}')


# class BackupManager():

#     @staticmethod
#     def get_date_folders(input_dir, date_format='%d-%m-%y') -> list[Any]:  # for backups
#         """Identify and return folders that match the given date format in their name."""
#         date_folders = []
#         for root, dirs, _ in os.walk(input_dir):
#             for dir_name in dirs:
#                 if is_date_string(dir_name, date_format):
#                     folder_path = os.path.join(root, dir_name)
#                     date_folders.append(folder_path)
#         return date_folders

#     @staticmethod
#     def copy_log_file(source_file, destination_file) -> LiteralString | Literal['Error: Log file not found']:
#         """The `copy_log_file` function copies a log file from a source file path to a destination file path
#         and returns a success message or an error message if the log file is not found."""
#         log_file = 'applogs.log'
#         log_file_path = source_file
#         backup_log_file_path = destination_file

#         if os.path.exists(log_file_path):
#             shutil.copy(log_file_path, backup_log_file_path)
#             logger.info(f'Log file copied: {log_file}')
#             return f'Log file copied: {log_file}'
#         else:
#             logger.warning(f'Log file not found: {log_file}')
#             return 'Error: Log file not found'

#     @staticmethod
#     def create_backups(input_dir, backup_dir, query_name) -> None:
#         """The `create_backups` function creates backups of specific file types in a given input directory,
#         using a unique filename based on the current timestamp and other parameters."""
#         SystemUtils.check_and_modify_permissions(backup_dir)

#         files = os.listdir(input_dir)
#         logger.info(f'input_dir {input_dir} - backup_dir {backup_dir}')
#         logger.info(f'Files found: {files}')
#         moved_files = []
#         AZDIAG_CSV_FILEPATH_STR = 'AzDiag.csv'
#         if AZDIAG_CSV_FILEPATH_STR in files:
#             csv_file_path = os.path.join(input_dir, AZDIAG_CSV_FILEPATH_STR)
#             df = pd.read_csv(csv_file_path, low_memory=False)
#             # df = pd.read_csv(os.path.join(input_dir, AZDIAG_CSV_FILEPATH_STR))

#             logger.info(f'First LocalTime found: {df["LocalTime"].iloc[0]}')
#             df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%d-%m-%y %H:%M:%S', errors='coerce')
#             first_timestamp = df['LocalTime'].iloc[0].strftime('%d-%m-%y_%I.%M%p')

#             logger.info(f'File loaded: {AZDIAG_CSV_FILEPATH_STR} which has {df.shape[0]} rows and {df.shape[1]} columns')
#             logger.info(f'Converted timestamp: {first_timestamp}')

#             extensions_to_copy = ['.csv', '.json', '.xlsx', '.log', '.txt']  # ! EXTENSION LIST
#             for file in files:
#                 filename, file_extension = os.path.splitext(file)
#                 if file_extension in extensions_to_copy:
#                     # ! Check if "FINAL" should be included in the filename.
#                     is_final = 'final' in filename.lower() or 'FINAL' in filename.lower()
#                     final_suffix = "_FINAL" if is_final else ""
#                     base_final_filename = f"{query_name}{final_suffix}_{first_timestamp}{file_extension}"
#                     final_filename = base_final_filename

#                     # ! Ensure the filename is unique within the backup directory.
#                     c_to = os.path.join(backup_dir, final_filename)
#                     counter = 1
#                     while os.path.exists(c_to):
#                         final_filename = f"{filename}_{counter}{final_suffix}_{first_timestamp}{file_extension}"
#                         c_to = os.path.join(backup_dir, final_filename)
#                         counter += 1

#                     shutil.copy(os.path.join(input_dir, file), c_to)
#                     logger.info(f'File copied: {final_filename}')
#                     moved_files.append(final_filename)

#         logger.info(f'Following files have been moved: {", ".join(moved_files)}')

#     @staticmethod
#     def store_old_backups(input_dir, days_threshold=7) -> None:
#         """The `store_old_backups` function takes an input directory and a days threshold, and archives and
#             deletes folders that are older than the threshold."""
#         backup_folders = [folder for folder, _, _ in os.walk(input_dir) if os.path.isdir(folder) and folder != input_dir]
#         logger.info(f'backup_folders: {backup_folders}')
#         date_format = '%d-%m-%y'
#         date_folders = BackupManager.get_date_folders(input_dir, date_format)

#         for folder in date_folders:
#             try:
#                 folder_date = datetime.strptime(os.path.basename(folder), date_format)
#                 current_date = datetime.now()
#                 date_difference = current_date - folder_date
#                 logger.info(f'Date difference for {folder}: {date_difference.days} days')

#                 if date_difference.days >= days_threshold:
#                     shutil.make_archive(folder, 'zip', folder)
#                     shutil.rmtree(folder)
#                     logger.info(f'Folder archived and deleted: {folder}')
#                 else:
#                     logger.info(f'Folder within threshold, skipping: {folder}')
#             except ValueError:
#                 logger.error(f'Folder name does not match date format and will be skipped: {folder}')

#     @staticmethod
#     def manage_backups(output_folder: str, query_name: str) -> None:
#         """The `manage_backups` function creates backup directories, performs backups, stores old backups, and
#         copies log files."""
#         kql_backup_dir = f'{output_folder}/{query_name}'
#         if Path(kql_backup_dir).is_dir():
#             logger.info(f'Backup dir exists: {kql_backup_dir}')
#         else:
#             logger.info(f'Backup dir does not exist: {kql_backup_dir}'
#                         f'\nCreating dir: {kql_backup_dir}')
#             Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
#         backup_dir = f'{kql_backup_dir}/{TODAY}'
#         if Path(backup_dir).is_dir():
#             logger.info(f'Backup dir exists: {backup_dir}')
#         else:
#             logger.info(f'Backup dir does not exist: {backup_dir}'
#                         f'\nCreating dir: {backup_dir}')
#             Path(backup_dir).mkdir(parents=True, exist_ok=True)
#         logger.info(f'Backup dir: {backup_dir}')
#         if not Path(backup_dir).is_dir() and not Path(kql_backup_dir).is_dir():  # ? NOTE
#             Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
#             Path(backup_dir).mkdir(parents=True, exist_ok=True)
#         # if not Path(backup_dir).is_dir():
#         #     if not Path(kql_backup_dir).is_dir():
#         #         Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
#         #         Path(backup_dir).mkdir(parents=True, exist_ok=True)
#         try:
#             SRC_LOG_FILE = f'{LOG_FOLDER}/applogs.log'
#             DEST_LOG_FILE = f'{backup_dir}/applogs.log'
#             BackupManager.create_backups(input_dir=output_folder, backup_dir=backup_dir, query_name=query_name)
#             logger.info(f'Backup completed: {backup_dir}')
#             BackupManager.store_old_backups(input_dir=output_folder, days_threshold=7)
#             logger.info(f'Old backups stored: {output_folder}')
#             BackupManager.copy_log_file(source_file=SRC_LOG_FILE, destination_file=DEST_LOG_FILE)
#             logger.info(f'copy_log_file: {SRC_LOG_FILE} to {DEST_LOG_FILE}')
#         except Exception as err:
#             logger.error(f'Error in manage_backups: {err}')
