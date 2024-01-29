# collector.py
import json
from datetime import datetime
from typing import Any, Tuple, List, Dict, Literal, TypedDict, cast, LiteralString
import logging
import inspect
import shutil
import sys
from pathlib import Path
import os
import stat
import re
from dateutil import parser
import pytz
import pandas as pd
import httpx
from convertTime import convert_utc_to_sydney
from configurator import excelConfigurator, regexConfigurator, load_kql_queries, setup_logging, load_config_from_file, jprint

OUTPUT_FOLDER: str = 'AZLOGS'
LOG_FOLDER: str = OUTPUT_FOLDER + '/LOG'
if not Path(OUTPUT_FOLDER).is_dir():
    Path(OUTPUT_FOLDER).mkdir()
if not Path(LOG_FOLDER).is_dir():
    Path(LOG_FOLDER).mkdir()
setup_logging()
kql_queries: Dict[str, Dict[str, str]] = load_kql_queries()

logger = logging.getLogger(__name__)
# #! Constants
KEYSEARCH_requestQuery_s: str = 'requestQuery_s'
KEYSEARCH_originalRequestUriWithArgs_s: str = 'originalRequestUriWithArgs_s'
KEYSEARCH_requestUri_s: str = 'requestUri_s'
CL_TimeGenerated: str = 'TimeGenerated'
CL_LocalTime: str = 'LocalTime'
DELALL_FEXT: Tuple[str, str, str, str] = ('.csv', '.xlsx', '.log', '.txt')


class ConfigLoader():

    @staticmethod
    def load_configurations():
        """The function "load_configurations" returns three objects: an excelConfigurator object, a
        regexConfigurator object, and the result of loading a configuration from a file."""
        config_keys = ['CL_INIT_ORDER', 'CL_DROPPED', 'CL_FINAL_ORDER', 'EXCLUSION_PAIRS']
        regex_keys = ['AZDIAG_PATTS', 'AZDIAG_STRINGS', 'MATCH_VALUE_TO_SPECIFIC_COLUMN']
        return excelConfigurator(config_keys), regexConfigurator(regex_keys), load_config_from_file()


class DataFrameUtilities():

    @staticmethod
    def find_duplicate_columns(df) -> Any:
        """The function `find_duplicate_columns` takes a DataFrame as input and returns a list of duplicate
        column names."""
        duplicate_columns = df.columns[df.columns.duplicated()]
        logger.info(f"Duplicate columns: {duplicate_columns.tolist()}")
        return duplicate_columns.tolist()


class TimeUtils():

    @staticmethod
    def get_current_time_info() -> dict[str, Any]:
        """The function `get_current_time_info` returns a dictionary containing the current time information in
        Sydney and in UTC."""
        utc_zone = pytz.utc
        sydney_zone = pytz.timezone('Australia/Sydney')
        utc_now = datetime.now(utc_zone)
        sydney_now = utc_now.astimezone(sydney_zone)
        return {
            'NOWINSYDNEY': sydney_now.strftime('%d-%m-%y %H:%M:%S'),
            'NOWINAZURE': utc_now,
            'TODAY': sydney_now.strftime('%d-%m-%y'),
            'NOWINSYDNEY_FILEFORMAT': sydney_now.strftime("%Y-%m-%d_%H-%M-%S"),
            'NOWINAZURE_FILEFORMAT': utc_now.strftime("%Y-%m-%d_%H-%M-%S")
        }

    @staticmethod
    def is_date_string(s, date_format) -> bool:  # for backups
        """Check if the string s matches the date_format."""
        try:
            datetime.strptime(s, date_format)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_time_format_AZURE(time_str) -> bool:
        """Validate if the input string matches Azure's expected date-time formats."""
        try:
            # Attempt to parse the datetime string
            parsed_time = parser.parse(time_str)
            logger.info(f"Validated and parsed time: {parsed_time.isoformat()}")
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_time_format(time_str) -> bool:
        """Validates if the time string includes a time unit (m, h, d, w). Returns True if valid, False otherwise."""
        return bool(re.match(r'^\d+[mhdw]$', time_str))


# ! TIME
time_info = TimeUtils.get_current_time_info()
NOWINSYDNEY: str = time_info['NOWINSYDNEY']
NOWINAZURE: str = time_info['NOWINAZURE']
TODAY: str = time_info['TODAY']
NOWINSYDNEY_FILEFORMAT: str = time_info['NOWINSYDNEY_FILEFORMAT']
NOWINAZURE_FILEFORMAT: str = time_info['NOWINAZURE_FILEFORMAT']
logger.info(f'Sydney Time {NOWINSYDNEY}')
logger.info(f'Sydney Time {NOWINAZURE}')


class APIManager():

    @staticmethod
    def get_logs_from_azure_analytics(query, headers, zen_endpoint) -> Any | dict:  # ! API call to get logs
        """The `get_logs_from_azure_analytics` function makes an API call to retrieve logs from Azure Analytics
        and returns the response as a JSON object."""

        try:
            logger.info(f"Log endpoint:{zen_endpoint}")
            json.dumps(headers)
            headers['Content-Type'] = 'application/json'
            response = httpx.post(url=zen_endpoint, data=query, headers=headers)
            response.raise_for_status()
            logger.info(f"Response URL: {response.url}")
            logger.info(f"Response Status Code: {response.status_code}")
            # logger.info(f'{response.text[:10]}')
            # logger.info(f'{response.text[-10:]}')
            # logger.info(f'Response Headers: {response.headers}')
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
        return {}

    @staticmethod
    def save_new_token(token_url, client_id, client_secret, resource_scope, token_file) -> Any | tuple[None, None]:
        """The `save_new_token` function sends a request to obtain a new token using client credentials, saves
        the token response to a file, and returns the token response."""
        try:
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret,
                'resource': resource_scope,
            }
            response = httpx.post(token_url, headers=headers, data=payload)
            response.raise_for_status()
            token_response = response.json()
            FileHandler.save_json(token_response, token_file)
            logger.info(f"New token RAW response saved to file {token_file}.")
            token_response = response.json()
            return token_response
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
        except Exception as e:
            logger.error(f"An error occurred: {str(object=e)}")
        return None, None

    # @staticmethod
    # def is_token_valid(token_info) -> bool:
    #     current_time = int(time.time())
    #     expires_on = int(token_info['expires_on'])
    #     return current_time < expires_on

    @staticmethod
    def fetch_and_save_api_data(
        token_url: str = '',
        client_id: str = '',
        client_secret: str = '',
        resource_scope: str = '',
        token_file_path: str = '',
        json_file_path: str = '',
        endpoint: str = ''
    ) -> tuple[Literal['Error in token'], Any | Literal['AZDIAG_IP1IP2TIMEAGO', 'AZDIAG_TIMEAGO', 'AZDIAG_TIMEBETWEEN', 'APPREQ_TIMEAGO', 'APPPAGE_TIMEAGO', 'APPBROWSER_TIMEAGO',
                                                        'APPSERVHTTPLogs_TIMEAGO', 'APPSERVIPSecTIMEAGO']] | tuple[Any | dict[
                                                            Any, Any], Any | Literal['AZDIAG_IP1IP2TIMEAGO', 'AZDIAG_TIMEAGO', 'AZDIAG_TIMEBETWEEN', 'APPREQ_TIMEAGO',
                                                                                     'APPPAGE_TIMEAGO', 'APPBROWSER_TIMEAGO', 'APPSERVHTTPLogs_TIMEAGO', 'APPSERVIPSecTIMEAGO']]:
        """The function fetches and saves API data using a token and query, and handles errors."""
        query_name, query_content = UserInputHandler.get_query_input()
        KQL = json.dumps({"query": query_content})
        logger.info(f'Active [yellow]Query Name:[/yellow] [red]{query_name}[/red]')
        logger.warning('Active [yellow]Query:[/yellow]')
        jprint(KQL)
        correct_query = input(f'Correct query? {query_name} Enter to continue, or type anything to exit...\n')
        if correct_query == '':
            logger.info(f'Proceed with: {query_name}')
        else:
            logger.info('Incorrect query, exiting...')

        token_info = APIManager.save_new_token(token_url, client_id, client_secret, resource_scope, token_file_path)
        if not token_info:
            logger.warning("Failed to refresh access token.")
            return 'Error in token', query_name
        try:
            token_key = token_info["access_token"]  # type: ignore
            headers = {'Authorization': f'Bearer {token_key}'}
            # logger.info(f"Headers: {headers}")
            query = KQL
            response = APIManager.get_logs_from_azure_analytics(query, headers, endpoint)
            if response:
                FileHandler.save_json(response, json_file_path)
                return response, query_name
            logger.error("Could not fetch data from API. Exiting.")
            sys.exit()
        except Exception as e:
            logger.error(f"e in fetch_and_save_api_data: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
            logger.error("Could not fetch data from API. Exiting.")
            sys.exit()


class FileHandler():

    @staticmethod
    def remove_files_from_folder(folder, file_extension) -> None:
        """Removes files with a specific file extension from a given folder.
        Args:folder (str): The path to the folder from which files will be removed.
            file_extension (str or Tuple[str]): The file extension(s) of the files to be removed."""
        try:
            files_to_remove = []
            for datafiles in os.listdir(folder):
                if datafiles.endswith(file_extension):
                    filepath = os.path.join(folder, datafiles)
                    if os.path.exists(filepath):
                        files_to_remove.append(datafiles)
                    else:
                        logger.warning(f"File not found: {datafiles}")
            for datafiles in files_to_remove:
                filepath = os.path.join(folder, datafiles)
                os.remove(filepath)
            logger.info(f"Removed {len(files_to_remove)} files from {folder}")
            jprint(files_to_remove)
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

    @staticmethod
    def saveTablesResponseToCSV(json_data, filename: str, exclusion_pairs: dict, log_file: str):
        """The `saveTablesResponseToCSV` function takes a JSON response containing tables, converts them into pandas DataFrames, drops any blank columns, and saves the resulting DataFrames as CSV files."""
        try:
            if 'tables' in json_data:
                for table in json_data['tables']:
                    logger.debug(f"Table name: {table['name']}")
                    df = pd.DataFrame(table['rows'], columns=[col['name'] for col in table['columns']])
                    DataFrameUtilities.find_duplicate_columns(df)
                    dataframe_row_length = len(df)
                    dataframe_column_length = len(df.columns)
                    logger.info(f'DataFrame Rows: {dataframe_row_length} - DataFrame Columns: {dataframe_column_length}')
                    df = DataFrameManipulator.processExclusionPairs(df, filename, exclusion_pairs, log_file)
                    logger.info("Proces exclusion pairs completed.")
                    if df is None:
                        logger.warning("DataFrame is empty. Exiting")
                        exit()
                    elif df.empty:
                        logger.warning("DataFrame is empty. Exiting")
                        exit()
                    if dataframe_row_length != len(df):
                        logger.info(f'New dataframe row and column count: {len(df)} x {len(df.columns)}')
                        logger.info(f'Difference afer exclusion pairs: {dataframe_row_length - len(df)} rows')
                    original_list_of_columns = df.columns.tolist()
                    original_list_of_columns_count = len(original_list_of_columns)
                    df.dropna(axis=1, how='all', inplace=True)
                    DataFrameUtilities.find_duplicate_columns(df)
                    logger.info(f'Original col count {original_list_of_columns_count} - After dropna col count {len(df.columns.tolist())}')
                    logger.debug(f'Dropped: {len(original_list_of_columns) - len(df.columns.tolist())} columns')
                    after_drop_list_of_columns = df.columns.tolist()

                    dropped_columns = "\n".join([col for col in original_list_of_columns if col not in after_drop_list_of_columns])
                    droppped_columns_count = len([col for col in original_list_of_columns if col not in after_drop_list_of_columns])
                    if dropped_columns:
                        logger.debug(f'Dropped columns:\n {dropped_columns} | Count: {droppped_columns_count}')
                    else:
                        logger.info('No cols dropped')
                    df.to_csv(filename, index=False, encoding='utf-8')
                    DataFrameUtilities.find_duplicate_columns(df)
                    return True
            else:
                logger.warning("No tables found in response.")
                return None
        except KeyError:
            logger.warning("No tables found in response.")
            return None

    @staticmethod
    def read_csv_add_LT_col_write_csv(input_file, source_col, dest_col, output_file, dropped_cols, initOrder_cols) -> None:
        """The `read_csv_add_LT_col_write_csv` function reads a CSV file, adds a new column based on a
        specified source column, drops specified columns, reorders the columns based on an initial order,
        checks for duplicate columns, and saves the modified DataFrame to a new CSV file."""
        try:
            df = pd.read_csv(input_file)
            logger.info(f"source_col: {source_col} | dest_col: {dest_col}")

            new_column_data = df[source_col].apply(convert_utc_to_sydney)
            if source_col not in df.columns:
                logger.warning(f"Column {source_col} not found in DataFrame.")
            else:
                df.insert(0, dest_col, new_column_data)

            # ! Drop specified columns if they exist
            df = df.drop(columns=[col for col in dropped_cols if col in df.columns], errors='ignore')

            # ! Ensure initial order columns are in the DataFrame and add them first to new_order
            new_order = [col for col in initOrder_cols if col in df.columns]

            # ! Add remaining columns that were not specified in initOrder_cols
            rest_columns = [col for col in df.columns if col not in new_order]
            new_order += rest_columns

            # ! Optional, for debugging
            duplicate_cols = DataFrameUtilities.find_duplicate_columns(df[new_order])
            if duplicate_cols:
                logger.info(f"Duplicate columns after reordering: {duplicate_cols}")
            else:
                logger.info("No duplicate columns after reordering.")

            df[new_order].to_csv(output_file, index=False)
            logger.info(f"DataFrame saved to CSV with new order: {output_file}")

        except Exception as e:
            logger.error(f'Error in processing data: {e}', exc_info=True, stack_info=True)

    @staticmethod
    def save_raw_logs(data: str, filename: str) -> None:
        """ Save raw logs to a file.
        Args:   data (str): The raw log data to be saved.
                filename (str): The name of the file to save the logs to."""
        try:
            logger.debug(f"Saving raw logs to file: {filename}")
            calling_function = None
            frame = inspect.currentframe()
            if frame is not None:
                if frame.f_back is not None:
                    calling_function = frame.f_back.f_code.co_name
                    logger.debug(f"Calling function: {calling_function}")
                else:
                    logger.warning("No calling function found.")
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"Calling function: {calling_function}\n")
                f.write(data)
        except IOError as e:
            logger.error(f"Error saving raw logs: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def save_removed_rows_to_raw_logs(removed_rows, column, pattern, filename) -> None:
        """The `save_removed_rows_to_raw_logs` function saves the removed rows from a DataFrame to a log file, including additional information for each row. Each removed row is appended to the log file, along with additional information if available.
        Args:   removed_rows (DataFrame): The DataFrame containing the removed rows.
                column (str): The name of the column from which the rows were removed.
                pattern (str): The pattern used to remove the rows.
                filename (str): The name of the log file to save the removed rows."""
        try:
            removed_count = len(removed_rows)
            log_data = f"Column: '{column}', Pattern: '{pattern}', Removed Rows Count: {removed_count}\n"  # ! Initialize the log data string
            for _, row in removed_rows.iterrows():
                log_data += "Removed Row:"  # ! Append the removed row to the log data
                log_data += "\n"
                log_data += f"{row[column]}"  # ! Append the removed row to the log data
                additional_columns = [KEYSEARCH_requestQuery_s, KEYSEARCH_originalRequestUriWithArgs_s, KEYSEARCH_requestUri_s]  # ! Define additional columns to capture
                additional_info = []
                for col in additional_columns:
                    if col in row:
                        additional_info.append(f"{col}: {row[col]}")  # ! Capture additional information from the row if the column exists
                if additional_info:
                    log_data += ", ".join(additional_info) + "\n"  # ! Append the additional information to the log data
            FileHandler.save_raw_logs(log_data, filename)  # ! Write the log data to the file
        except Exception as e:
            logger.error(f'E in save_removed_rows_to_raw_logs: {e}', exc_info=True, stack_info=True)


class DataFrameManipulator():

    @staticmethod
    def processExclusionPairs(df: pd.DataFrame, filename: str, exclusion_pairs, log_file: str):
        """The `processExclusionPairs` function takes a DataFrame, a filename, a dictionary of exclusion pairs,
        and a log file as input, and processes the exclusion pairs by removing rows from the DataFrame that
        match the specified values, logging the removed rows, and saving the potentially modified DataFrame
        to a CSV file."""
        try:
            logger.info("[green]##########Processing exclusion pairs##########[/green]")
            for column, values_to_exclude in exclusion_pairs.items():
                if column in df.columns:
                    logger.info(f"Column: {column} found in DataFrame. Values to exclude: {values_to_exclude}")
                    filter_condition = df[column].isin(values_to_exclude)
                    # ! Capture the rows that will be removed
                    removed_rows = df[filter_condition].copy()  # ? Use .copy() to avoid SettingWithCopyWarning
                    # ? Log the removed rows if there are any
                    if not removed_rows.empty:
                        pattern = f"Exclusion: {values_to_exclude}"
                        FileHandler.save_removed_rows_to_raw_logs(removed_rows, column, pattern, log_file)
                        logger.info(f"Removed rows based on column '{column}' and pattern '{pattern}' have been logged to {log_file}")
                    # ? update df to exclude the rows
                    df = df[~filter_condition]
                    if df.empty:
                        logger.info("DataFrame is empty after exclusion.")
                        break  # ! Exit the loop if df is empty
                    # ! Identify columns that may be dropped due to becoming entirely NaN
                    columns_before = set(df.columns)
                    df.dropna(axis=1, how='all', inplace=True)
                    columns_after = set(df.columns)
                    columns_dropped = columns_before - columns_after
                    if columns_dropped:
                        logger.debug(f"Columns dropped because they became empty after exclusion: {columns_dropped}")
                    else:
                        logger.warning("No columns were dropped after exclusion.")
            # ! After processing, save the potentially modified df
            if not df.empty:
                df.to_csv(filename, index=False)
                logger.debug(f"Modified DataFrame saved to {filename}")
            else:
                logger.warning("DataFrame is empty after exclusions. Skipping saving to CSV.")
            return df
        except Exception as e:
            logger.error(f"Error in exclusion processing: {e}", exc_info=True)
            return df

    @staticmethod
    def remove_patterns(dataframe, extraction_file, regex, string, key_col_to_val_patts) -> Any:
        """The `remove_patterns` function takes a dataframe, extraction file, regex patterns, string patterns,
        and key-value patterns as input, removes the specified patterns from the dataframe, and returns the
        modified dataframe."""
        try:
            logger.debug('[yellow]##########Starting remove_patterns##########[/yellow]')
            df = dataframe
            columns_to_search = [KEYSEARCH_originalRequestUriWithArgs_s, KEYSEARCH_requestQuery_s, KEYSEARCH_requestUri_s]
            total_removed = 0
            output = ""
            df_columns = df.columns.tolist()
            df_total_columns = len(df.columns)
            df_total_rows = len(df)
            logger.info(f'DF Total Columns: {df_total_columns} - Total Rows: {df_total_rows}')
            logger.debug(f'df_columns {df_columns}')
            regex_json_formatted = json.dumps(regex, indent=4)
            string_json_formatted = json.dumps(string, indent=4)
            logger.debug('\nRegex patterns:')
            logger.debug(regex_json_formatted)
            logger.debug('\nString patterns:')
            logger.debug(string_json_formatted)
            df = DataFrameManipulator.remove_regex_patterns(df, regex, string, columns_to_search, extraction_file)
            df = DataFrameManipulator.remove_key_value_patterns(df, key_col_to_val_patts, extraction_file)
            output += f"\nTotal rows Total Removed: {total_removed}:\nRemaining Rows: {len(df)}\n"
            return df
        except Exception as e:
            logger.error(f'Error in remove_patterns: {e}', exc_info=True, stack_info=True)

    @staticmethod
    def remove_regex_patterns(df, regex, string, columns_to_search, extraction_file) -> Any:
        """The `remove_regex_patterns` function removes rows from a DataFrame that match specified regex
        patterns in specified columns and saves the removed rows to a file."""
        logger.debug('[yellow]##########Starting remove_regex_patterns##########[/yellow]')
        total_removed = 0
        output = ""
        for pattern in regex + string:
            logger.debug(f'Looking for pattern: {pattern} in columns: {columns_to_search}')
            for column in columns_to_search:
                if column not in df.columns:
                    logger.debug(f'Column {column} not found in the dataframe.')
                    continue
                pattern_filter = df[column].str.contains(pattern, na=False, regex=True)
                removed_rows = df[pattern_filter]

                logger.debug(f'REGEX Pat {pattern} - Col {column} - Removed Rows {len(removed_rows)}')
                logger.debug(f'Rowdata: removed_rows[column]: {removed_rows[column]}')

                if not removed_rows.empty:
                    FileHandler.save_removed_rows_to_raw_logs(removed_rows, column, pattern, extraction_file)
                    output += f"Pattern: {pattern} - Column: {column} - Removed Count: {len(removed_rows)}"
                    removed_rows_text = removed_rows[column]
                    output += str(removed_rows_text)
                df = df[~pattern_filter]
                removed_count = len(removed_rows)
                total_removed += removed_count
        logger.debug('[yellow]##########End of remove_regex_patterns##########[/yellow]')
        return df

    @staticmethod
    def remove_key_value_patterns(df, key_col_to_val_patts, extraction_file):
        """The `remove_key_value_patterns` function removes rows from a dataframe based on specified key-value patterns in specific columns and logs the removed rows."""
        logger.debug('[yellow]##########Key Value Removals Started##########[/yellow]')
        total_removed = 0
        output = ""
        for column, patterns in key_col_to_val_patts.items():
            removed_count = 0  # ! Initialize removed_count at the start of the loop
            logger.debug(f'Looking for column: {column} patterns {patterns} key_cl.items {key_col_to_val_patts.items()}')
            if column not in df.columns:
                logger.warning(f'Column {column} not found in the dataframe.')
                continue
            logger.debug(f"Processing column: {column} with patterns: {patterns}")
            for pattern in patterns:
                pattern_filter = df[column].str.contains(pattern, na=False, regex=True)
                removed_rows = df[pattern_filter]
                if not removed_rows.empty:
                    removed_count = len(removed_rows)
                    logger.debug('###KEY_CL_TO_PATT Removals###')
                    logger.debug(f'Pat {pattern} - Col {column} - Removed Rows {removed_count}')
                    FileHandler.save_removed_rows_to_raw_logs(removed_rows, column, pattern, extraction_file)
                    df = df[~pattern_filter]
                    total_removed += removed_count
                if removed_count > 0:
                    output += f"\nMatch:Pattern: {pattern} \nRemoved Count: {removed_count} \nColumn: {column}\n"
                else:
                    output += f"\nNo match:\nPattern: {pattern} \nColumn: {column}\n"
        logger.debug('[yellow]##########Key Value Removals Completed##########[/yellow]')
        return df


class ExcelManager():

    @staticmethod
    def excelCreate(input_csv_file, output_excel_file, extraction_log_file, regex_patterns, string_patterns, col_to_val_patterns) -> None:
        """The function `excelCreate` reads a CSV file, removes specified patterns from the data, drops columns with all NaN values, and creates an Excel file with the formatted data."""
        try:
            df = pd.read_csv(input_csv_file)
            df = DataFrameManipulator.remove_patterns(dataframe=df,
                                                      extraction_file=extraction_log_file,
                                                      regex=regex_patterns,
                                                      string=string_patterns,
                                                      key_col_to_val_patts=col_to_val_patterns)

            logger.info(f'Captured Logs: {extraction_log_file} - Rows: {df.shape[0]} - Col: {df.shape[1]}')
            original_list_of_columns = df.columns.tolist()
            df.dropna(axis=1, how='all', inplace=True)  # ! Remove columns with all NaN values
            logger.info(
                f'Original count: {len(original_list_of_columns)} - After drop: {len(df.columns.tolist())} - Dropped: {len(original_list_of_columns) - len(df.columns.tolist())} blank columns'
            )

            if not df.empty:
                with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:  # pylint: disable=abstract-class-instantiated
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                    ExcelManager.format_excel_file(writer, df)
                    logger.debug(f'Excel formatted and created: {output_excel_file}')
            else:
                logger.warning('Dataframe is empty')
        except Exception as e:
            logger.error(f'E in create_excel: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def format_excel_file(writer, df) -> None:
        """The `format_excel_file` function formats an Excel file by setting the font, font size, and column width for each column in a given DataFrame."""
        try:
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            cell_format = workbook.add_format({'font_name': 'Open Sans', 'font_size': 8})
            worksheet.set_column('A:ZZ', None, cell_format)
            for column in df:
                column_length = max(df[column].astype(str).apply(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.set_column(col_idx, col_idx, column_length, cell_format)
        except Exception as e:
            logger.error(f'Error in format_excel_file: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def createFinalExcel(input_file: str, output_file: str, extraction_log_file: str, columns_to_be_dropped: List[str], final_column_order: List[str],
                         regex_patterns: Dict[str, str], string_patterns: Dict[str, str], col_to_val_patterns: Dict[str, str]) -> None:
        """The `createFinalExcel` function takes an input file, applies various extraction patterns and
        filters, and creates a final Excel file with specified column order and dropped columns."""

        def concatenate_values(row: pd.Series, columns: List[str], include_col_names: bool = False) -> str:
            """The function `concatenate_values` takes a row of a pandas DataFrame, a list of column names, and an
            optional flag to include column names, and returns a string concatenating the non-null values of the
            specified columns."""
            if include_col_names:
                return ', '.join([f'{col}: "{row[col]}"' for col in columns if pd.notna(row[col])])
            return ', '.join([f'"{row[col]}"' for col in columns if pd.notna(row[col])])

        def drop_and_create_columns(df: pd.DataFrame, new_columns: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
            """The function `drop_and_create_columns` takes a DataFrame and a dictionary of new columns as input,
            creates new columns based on existing columns in the DataFrame, drops columns that were used to
            create the new columns, and returns the modified DataFrame."""
            logger.info('[red]########### columns_to_drop started ###########[/red]')
            for col_name, info in new_columns.items():
                existing_cols = [col for col in info['cols'] if col in df.columns]
                missing_cols = [col for col in info['cols'] if col not in df.columns]
                if existing_cols:
                    # ! Create new column by concatenating values from existing columns
                    df[col_name] = df.apply(lambda row, cols=existing_cols: concatenate_values(row, cols, info['include_names']), axis=1)  # pylint: disable=W0640
                    logger.info(f'New column "{col_name}" created with values from columns: {existing_cols} - Total rows: {df.shape[0]}')
                    if col_name != 'Qs':
                        columns_to_drop_due_to_concat.extend(existing_cols)
                    else:
                        logger.info(f'Columns kept: {existing_cols}')
                if missing_cols:
                    logger.debug(f'Columns not found in "{col_name}": {missing_cols}')
            if columns_to_drop_due_to_concat:
                # ! Drop columns that were used to create new columns
                df.drop(columns=columns_to_drop_due_to_concat, inplace=True)
                dropped_columns = available_columns_before_drop - set(df.columns)
                logger.debug(f'Columns dropped from concat: {columns_to_drop_due_to_concat}')
                if dropped_columns:
                    logger.debug("Columns dropped from concat:")
                    for column in dropped_columns:
                        logger.debug(column)
                else:
                    logger.info("No columns were dropped.")
            logger.info('[red]########### columns_to_drop  from concat completed ###########[/red]')
            logger.info(f'Columns dropped.\nFinal DF: {df.shape[0]} rows {df.shape[1]} columns')
            logger.info(f'[green]Columns to be dropped started: {columns_to_be_dropped}[/green]')
            available_columns_before_drop2 = set(df.columns)
            cols_to_drop_from_func_def = [column for column in columns_to_be_dropped if column in df.columns]
            # ! Drop columns specified in the function definition
            if cols_to_drop_from_func_def:
                df.drop(columns=cols_to_drop_from_func_def, inplace=True)
                dropped_columns2 = available_columns_before_drop2 - set(df.columns)
                if dropped_columns2:
                    logger.info("Columns dropped:")
                    for column in dropped_columns2:
                        logger.info(column)
                logger.info(f'Columns dropped 2.\nFinal DF: {df.shape[0]} rows {df.shape[1]} columns')
                logger.info('[green]Columns to be dropped ended: [/green]')
            return df

        def apply_final_column_order_and_save(df: pd.DataFrame, final_column_order: List[str], output_file: str, available_columns_before_drop: set) -> None:
            """The function `apply_final_column_order_and_save` takes a DataFrame, a list of column names, an
            output file path, and a set of available columns as input, applies the specified column order to the
            DataFrame, saves the DataFrame to an Excel file, and logs the process."""
            final_order_logs = []
            final_order_logs += ['Final order started']
            final_order_logs += ['actual_final_order']
            actual_final_order = [col for col in final_column_order if col in df.columns]
            final_order_logs += [str(actual_final_order)]
            final_order_logs += ['specified_columns']
            specified_columns = set(actual_final_order)
            final_order_logs += [str(specified_columns)]
            final_order_logs += ['unspecified_columns']
            unspecified_columns = available_columns_before_drop - specified_columns
            final_order_logs += [str(unspecified_columns)]
            final_order_logs += ['unspecified_columns']
            unspecified_columns = unspecified_columns - set(columns_to_be_dropped)
            final_order_logs += [str(unspecified_columns)]
            final_order_logs += ['final_columns_list']

            final_columns_list = actual_final_order + [col for col in list(unspecified_columns) if col in df.columns]
            final_order_logs += [str(final_columns_list)]
            FINAL_ORDER_LOG_FILE = LOG_FOLDER + 'finalOrder.log'
            with open(FINAL_ORDER_LOG_FILE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(final_order_logs))
                logger.info(f'Final order log file created: {FINAL_ORDER_LOG_FILE}')
            logger.info(f'Final columns list: {final_columns_list}')
            try:
                # ! Subset df with the final columns list
                df = df[final_columns_list]

                # ! Proceed with saving the file
                df.to_excel(output_file, index=False)
                logger.info(f'Excel file created: {output_file}')
            except KeyError as e:
                logger.error(f"Error in applying final column order: {e}")
                # ! Handle the error or perform additional logging as needed

        df: pd.DataFrame = pd.read_excel(input_file)
        # ! Section 1
        columns_to_drop_due_to_concat: List[str] = []
        available_columns_before_drop = set(df.columns)
        df_total_rows: int = len(df)
        df_total_columns: int = len(df.columns)
        new_columns: Dict[str, Dict[str, Any]] = {
            'Qs': {
                'cols': ['requestQuery_s', 'originalRequestUriWithArgs_s', 'requestUri_s'],
                'include_names': True
            },
            'IPs': {
                'cols': ['clientIp_s', 'clientIP_s'],
                'include_names': False
            },
            'Host': {
                'cols': ['host_s', 'hostname_s', 'originalHost_s'],
                'include_names': False
            },
            'messagesD': {
                'cols': ['Message', 'action_s'],
                'include_names': True
            },
            'allD': {
                'cols': ['details_message_s', 'details_data_s', 'details_file_s', 'details_line_s'],
                'include_names': True
            },
            'rulesD': {
                'cols': ['engine_s', 'ruleSetVersion_s', 'ruleGroup_s', 'ruleId_s', 'ruleSetType_s', 'policyId_s', 'policyScope_s', 'policyScopeName_s'],
                'include_names': True
            },
            'serverD': {
                'cols': ['backendSettingName_s', 'noOfConnectionRequests_d', 'serverRouted_s', 'httpVersion_s'],
                'include_names': True
            },
            'Method_Type': {
                'cols': ['httpMethod_s', 'contentType_s'],
                'include_names': False
            },
            'TimeGenerated_eventD': {
                'cols': [
                    'eventTimestamp_s', 'errorMessage_s', 'errorCount_d', 'Environment_s', 'Region_s', 'ActivityName_s', 'operationalResult_s', 'ActivityId_g', 'ScaleUnit_s',
                    'NamespaceName_s'
                ],
                'include_names': True
            },
        }
        logger.debug('[red]########## Final Excel Creation Started #########[/red]')
        logger.debug(f'Dataframe loaded from file: {input_file}')
        logger.debug(f'Total Columns: {df_total_columns} - Total Rows: {df_total_rows}')
        df = DataFrameManipulator.remove_patterns(dataframe=df,
                                                  extraction_file=extraction_log_file,
                                                  regex=regex_patterns,
                                                  string=string_patterns,
                                                  key_col_to_val_patts=col_to_val_patterns)
        logger.info('Manipulations.remove_patterns completed')
        # ! Section 2
        drop_and_create_columns(df, new_columns)
        # ! Section 3
        apply_final_column_order_and_save(df, final_column_order, output_file, available_columns_before_drop)


class ExcelConfigurations(TypedDict):
    CL_INIT_ORDER: List[str]
    CL_DROPPED: List[str]
    CL_FINAL_ORDER: List[str]
    EXCLUSION_PAIRS: Dict[str, List[str]]


class RegexConfigurations(TypedDict):
    AZDIAG_PATTS: List[str]
    AZDIAG_STRINGS: List[str]
    MATCH_VALUE_TO_SPECIFIC_COLUMN: Dict[str, List[str]]


class AppConfigurations(TypedDict):
    TENANT_ID: str
    CLIENT_ID: str
    CLIENT_SECRET: str
    ZEN_WORKSPACE_ID: str
    AZDIAG_SCOPE: str
    AZLOG_ENDPOINT: str
    TOKEN_FILEPATH_STR: str
    AZDIAG_JSON_FILEPATH_STR: str
    AZDIAG_CSV_FILEPATH_STR: str
    AZDIAG_EXCEL_FILEPATH: Path
    AZDIAG_EXCEL_FINAL_FILEPATH: Path
    AZDIAG_EXTRACTIONLOGS_FILEPATH_STR: str
    TOKEN_URL: str
    AZLOG_ZEN_ENDPOINT: str


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


class QueryFormatter():

    @staticmethod
    def format_query(query_name, ip1=None, ip2=None, timeago=None, iplist=None, start_t=None, end_t=None) -> str:  # pylint: disable=unused-argument # noqa: W0613
        """Formats a query based on the given parameters.
        Parameters:
        - query_name: The name of the query.
        - ip1: The value for IP1 (optional).
        - ip2: The value for IP2 (optional).
        - timeago: The value for TIMEAGO (optional).
        - iplist: The list of IPs (optional).
        - start_t: The start time (optional).
        - end_t: The end time (optional).
        Returns:
        - str: The formatted query."""
        query_param_map = {
            "AZDIAG_IP1IP2TIMEAGO": {
                "IP1": ip1,
                "IP2": ip2,
                "TIME": timeago
            },
            "AZDIAG_TIMEBETWEEN": {
                "STARTTIME": start_t,
                "ENDTIME": end_t
            },
            "AZDIAG_TIMEAGO": {
                "TIME": timeago
            },
            "APPREQ_TIMEAGO": {
                "TIME": timeago
            },
            "APPPAGE_TIMEAGO": {
                "TIME": timeago
            },
            "APPBROWSER_TIMEAGO": {
                "TIME": timeago
            },
            "APPSERVHTTPLogs_TIMEAGO": {
                "TIME": timeago
            },
            "APPSERVIPSecTIMEAGO": {
                "TIME": timeago
            },
        }

        formatted_query_name = f"RAW_{query_name}"
        if formatted_query_name in raw_kql_queries:
            query_template = raw_kql_queries[formatted_query_name]
            query_params = query_param_map.get(query_name, {})
            return query_template.format(**query_params)
        logger.error(f'Active [yellow]Query Name:[/yellow] [red]{query_name}[/red] not found')
        return 'Not Found'


class BackupManager():

    @staticmethod
    def get_date_folders(input_dir, date_format='%d-%m-%y') -> list[Any]:  # for backups
        """Identify and return folders that match the given date format in their name."""
        date_folders = []
        for root, dirs, _ in os.walk(input_dir):
            for dir_name in dirs:
                if TimeUtils.is_date_string(dir_name, date_format):
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
        extensions_to_copy = ['.csv', '.json', '.xlsx', '.log', '.txt']
        files = os.listdir(input_dir)
        logger.info(f'input_dir {input_dir} - backup_dir {backup_dir}')
        logger.info(f'Files found: {files}')
        moved_files = []
        AZDIAG_CSV_FILEPATH_STR = 'AzDiag.csv'
        if AZDIAG_CSV_FILEPATH_STR in files:
            df = pd.read_csv(os.path.join(input_dir, AZDIAG_CSV_FILEPATH_STR))
            logger.info(f'First LocalTime found: {df["LocalTime"].iloc[0]}')
            df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%d-%m-%y %H:%M:%S', errors='coerce')
            first_timestamp = df['LocalTime'].iloc[0].strftime('%d-%m-%y_%I.%M%p')

            logger.info(f'File loaded: {AZDIAG_CSV_FILEPATH_STR} which has {df.shape[0]} rows and {df.shape[1]} columns')
            logger.info(f'Converted timestamp: {first_timestamp}')

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

        if not Path(backup_dir).is_dir():
            if not Path(kql_backup_dir).is_dir():
                Path(kql_backup_dir).mkdir(parents=True, exist_ok=True)
                Path(backup_dir).mkdir(parents=True, exist_ok=True)
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


class DataProcessor():

    @staticmethod
    def process_data(AZDIAG_JSON_FILEPATH_STR, AZDIAG_CSV_FILEPATH_STR, AZDIAG_EXTRACTIONLOGS_FILEPATH_STR, AZDIAG_EXCEL_FILEPATH, AZDIAG_EXCEL_FINAL_FILEPATH, CL_INIT_ORDER,
                     CL_DROPPED, AZDIAG_REGEX, AZDIAG_STRINGS, MATCH_VALUE_TO_SPECIFIC_COLUMN, EXCLUSION_PAIRS, CL_FINAL_ORDER) -> None:
        """Process data by reading JSON data, saving tables as CSV files, adding a local time column, creating an Excel file, and creating a final Excel file.
        Args: AZDIAG_JSON_FILEPATH_STR (str): File path of the JSON data.
            AZDIAG_CSV_FILEPATH_STR (str): File path to save the CSV file.
            AZDIAG_EXTRACTIONLOGS_FILEPATH_STR (str): File path to save extraction logs.
            AZDIAG_EXCEL_FILEPATH (str): File path to save the initial Excel file.
            AZDIAG_EXCEL_FINAL_FILEPATH (str): File path to save the final Excel file.
            CL_INIT_ORDER (List[str]): List of column names for initial column order.
            CL_DROPPED (List[str]): List of column names to be dropped.
            AZDIAG_REGEX (Dict[str, str]): Dictionary of regex patterns.
            AZDIAG_STRINGS (Dict[str, str]): Dictionary of string patterns.
            MATCH_VALUE_TO_SPECIFIC_COLUMN (Dict[str, str]): Dictionary of column-value patterns.
            EXCLUSION_PAIRS (Dict[str, str]): Dictionary of exclusion pairs.
            CL_FINAL_ORDER (List[str]): List of column names for final column order."""
        try:
            json_data = FileHandler.read_json(filename=AZDIAG_JSON_FILEPATH_STR)
            if 'tables' in json_data:
                saveData = FileHandler.saveTablesResponseToCSV(json_data=json_data,
                                                               filename=AZDIAG_CSV_FILEPATH_STR,
                                                               exclusion_pairs=EXCLUSION_PAIRS,
                                                               log_file=AZDIAG_EXTRACTIONLOGS_FILEPATH_STR)
                if saveData:
                    FileHandler.read_csv_add_LT_col_write_csv(input_file=AZDIAG_CSV_FILEPATH_STR,
                                                              source_col=CL_TimeGenerated,
                                                              dest_col=CL_LocalTime,
                                                              output_file=AZDIAG_CSV_FILEPATH_STR,
                                                              dropped_cols=CL_DROPPED,
                                                              initOrder_cols=CL_INIT_ORDER)
                    logger.debug(f'LT column added to {AZDIAG_CSV_FILEPATH_STR}')

                    logger.debug(f'excelCreate: Input {AZDIAG_CSV_FILEPATH_STR} | Output {AZDIAG_EXCEL_FILEPATH}')
                    ExcelManager.excelCreate(input_csv_file=AZDIAG_CSV_FILEPATH_STR,
                                             output_excel_file=AZDIAG_EXCEL_FILEPATH,
                                             extraction_log_file=AZDIAG_EXTRACTIONLOGS_FILEPATH_STR,
                                             regex_patterns=AZDIAG_REGEX,
                                             string_patterns=AZDIAG_STRINGS,
                                             col_to_val_patterns=MATCH_VALUE_TO_SPECIFIC_COLUMN)
                    logger.debug(f'excelCreate: Input {AZDIAG_CSV_FILEPATH_STR} | Output {AZDIAG_EXCEL_FILEPATH}')

                    try:
                        logger.debug(f'createFinalExcel: Input {AZDIAG_EXCEL_FILEPATH} | Output {AZDIAG_EXCEL_FINAL_FILEPATH}')
                        ExcelManager.createFinalExcel(input_file=AZDIAG_EXCEL_FILEPATH,
                                                      output_file=AZDIAG_EXCEL_FINAL_FILEPATH,
                                                      extraction_log_file=AZDIAG_EXTRACTIONLOGS_FILEPATH_STR,
                                                      columns_to_be_dropped=CL_DROPPED,
                                                      final_column_order=CL_FINAL_ORDER,
                                                      regex_patterns=AZDIAG_REGEX,
                                                      string_patterns=AZDIAG_STRINGS,
                                                      col_to_val_patterns=MATCH_VALUE_TO_SPECIFIC_COLUMN)
                        logger.debug(f'createFinalExcel: Input {AZDIAG_EXCEL_FILEPATH} | Output {AZDIAG_EXCEL_FINAL_FILEPATH}')
                    except Exception as e:
                        logger.error(f'E in createFinalExcel: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
        except Exception as e:
            logger.error(f'E in process_local_data: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)


class UserInputHandler():

    @staticmethod
    def select_query(
    ) -> Any | Literal['AZDIAG_IP1IP2TIMEAGO', 'AZDIAG_TIMEAGO', 'APPREQ_TIMEAGO', 'APPPAGE_TIMEAGO', 'APPBROWSER_TIMEAGO', 'APPSERVHTTPLogs_TIMEAGO', 'APPSERVIPSecTIMEAGO']:
        """Selects a query from a list of options and returns the chosen query."""
        logger.info('Select query')
        logger.info('1. AZDIAG_IP1IP2TIMEAGO')
        logger.info('2. AZDIAG_TIMEAGO')
        logger.info('3. AZDIAG_TIMEBETWEEN')
        logger.info('4. APPREQ_TIMEAGO')
        logger.info('5. APPPAGE_TIMEAGO')
        logger.info('6. APPBROWSER_TIMEAGO')
        logger.info('7. APPSERVHTTPLogs_TIMEAGO')
        logger.info('8. APPSERVIPSecTIMEAGO')
        logger.info('9. Exit')
        query = input('Select query: ')
        if query == '1':
            return a
        elif query == '2':
            return a1
        elif query == '3':
            return a2
        elif query == '4':
            return a3
        elif query == '5':
            return a4
        elif query == '6':
            return a5
        elif query == '7':
            return a6
        elif query == '8':
            return a7
        elif query == '9':
            sys.exit()
        logger.info('Wrong query, try again')
        return UserInputHandler.select_query()

    @staticmethod
    def input_datetime_with_validation(prompt) -> str:
        """Prompt the user for a datetime input and validate or adjust it to a proper format."""
        while True:
            input_time = input(prompt)
            if TimeUtils.validate_time_format_AZURE(input_time):
                return parser.parse(input_time).isoformat()
            else:
                logger.info("Invalid format. Please use '2024-01-28T17:57:38Z' or '2024-01-28 17:55:38.994534+00:00' or '2024-01-28T17:57:35.703683Z'.")

    @staticmethod
    def get_query_input() -> tuple[
        Any | Literal[
            'AZDIAG_IP1IP2TIMEAGO',
            'AZDIAG_TIMEAGO',
            'AZDIAG_TIMEBETWEEN',
            'APPREQ_TIMEAGO',
            'APPPAGE_TIMEAGO',
            'APPBROWSER_TIMEAGO',  # ! Here
            'APPSERVHTTPLogs_TIMEAGO',
            'APPSERVIPSecTIMEAGO'],
        str]:  # ? 1
        """Get user input for selecting and formatting a query.
        Returns:tuple: A tuple containing the selected query name and the formatted query content."""
        query_choice = UserInputHandler.select_query()
        query_name = query_choice
        ip1 = ip2 = None
        if query_name == "AZDIAG_IP1IP2TIMEAGO":
            ip1 = input('Enter value for IP1: ')
            ip2 = input('Enter value for IP2: ')
            timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            while not TimeUtils.validate_time_format(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, ip1=ip1, ip2=ip2, timeago=timeago)
        elif query_name == "AZDIAG_TIMEBETWEEN":
            STARTTIME = UserInputHandler.input_datetime_with_validation('Start time: ')
            ENDTIME = UserInputHandler.input_datetime_with_validation('End time: ')
            query_content = QueryFormatter.format_query(query_name, start_t=STARTTIME, end_t=ENDTIME)

        else:
            timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            while not TimeUtils.validate_time_format(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago)
        return query_name, query_content


raw_kql_queries = {
    "RAW_AZDIAG_IP1IP2TIMEAGO": kql_queries["F_AzDiag_IP1IP2TIMEAGO"]["query"],
    "RAW_AZDIAG_TIMEBETWEEN": kql_queries["F_AzDiag_TIMEBETWEEN"]["query"],
    "RAW_AZDIAG_TIMEAGO": kql_queries["F_AzDiag_TIMEAGO"]["query"],
    "RAW_APPREQ_TIMEAGO": kql_queries["F_AppReq_TIMEAGO"]["query"],
    "RAW_APPPAGE_TIMEAGO": kql_queries["F_AppPage_TIMEAGO"]["query"],
    "RAW_APPBROWSER_TIMEAGO": kql_queries["F_AppBrowser_TIMEAGO"]["query"],
    "RAW_APPSERVHTTPLogs_TIMEAGO": kql_queries["F_AppServHTTPLogs_TIMEAGO"]["query"],
    "RAW_APPSERVIPSecTIMEAGO": kql_queries["F_AppServIPSecTIMEAGO"]["query"],
}
# # ? IPs
# KQL_IPLIST = ['14.161.17.210', '115.78.224.106']
# KQL_IP1 = '14.161.17.210'
# KQL_IP2 = '115.78.224.106'
# # ! TIME
# KQL_TIMEAGO_TIME = '2m'
# STARTTIME = '2024-01-25T06:05:00Z'
# ENDTIME = '2024-01-25T06:15:00Z'
# # endTime = '2024-01-25T06:14:34.0000000'
# ! OPTIONS
a = 'AZDIAG_IP1IP2TIMEAGO'
a1 = 'AZDIAG_TIMEAGO'
a2 = 'AZDIAG_TIMEBETWEEN'
a3 = 'APPREQ_TIMEAGO'
a4 = 'APPPAGE_TIMEAGO'
a5 = 'APPBROWSER_TIMEAGO'
a6 = 'APPSERVHTTPLogs_TIMEAGO'
a7 = 'APPSERVIPSecTIMEAGO'


def main() -> None:
    """The main function loads configurations, processes data from either a local file or an API, and
    manages backups."""
    excel_config, regex_config, az_config = ConfigLoader.load_configurations()
    excel_config = cast(ExcelConfigurations, excel_config)
    regex_config = cast(RegexConfigurations, regex_config)
    az_config = cast(AppConfigurations, az_config)

    # # ! Columns
    CL_INIT_ORDER: List[str] = excel_config['CL_INIT_ORDER']
    CL_DROPPED: List[str] = excel_config['CL_DROPPED']
    CL_FINAL_ORDER: List[str] = excel_config['CL_FINAL_ORDER']
    EXCLUSION_PAIRS: Dict[str, List[str]] = excel_config['EXCLUSION_PAIRS']
    AZDIAG_REGEX: List[str] = regex_config['AZDIAG_PATTS']
    AZDIAG_STRINGS: List[str] = regex_config['AZDIAG_STRINGS']
    MATCH_VALUE_TO_SPECIFIC_COLUMN: Dict[str, List[str]] = regex_config['MATCH_VALUE_TO_SPECIFIC_COLUMN']

    # # ! Configs
    TENANT_ID: str = az_config['TENANT_ID']
    CLIENT_ID: str = az_config['CLIENT_ID']
    CLIENT_SECRET: str = az_config['CLIENT_SECRET']
    ZEN_WORKSPACE_ID: str = az_config['ZEN_WORKSPACE_ID']
    AZDIAG_SCOPE: str = az_config['AZDIAG_SCOPE']
    AZLOG_ENDPOINT: str = az_config['AZLOG_ENDPOINT']
    TOKEN_FILEPATH_STR: str = az_config['TOKEN_FILEPATH_STR']
    AZDIAG_JSON_FILEPATH_STR: str = az_config['AZDIAG_JSON_FILEPATH_STR']
    AZDIAG_CSV_FILEPATH_STR: str = az_config['AZDIAG_CSV_FILEPATH_STR']
    AZDIAG_EXCEL_FILEPATH: Path = Path(az_config['AZDIAG_EXCEL_FILEPATH'])
    AZDIAG_EXCEL_FINAL_FILEPATH: Path = Path(az_config['AZDIAG_EXCEL_FINAL_FILEPATH'])
    AZDIAG_EXTRACTIONLOGS_FILEPATH_STR: str = az_config['AZDIAG_EXTRACTIONLOGS_FILEPATH_STR']
    TOKEN_URL: str = f'https://login.microsoftonline.com/{TENANT_ID}/oauth2/token'
    AZLOG_ZEN_ENDPOINT: str = f'{AZLOG_ENDPOINT}{ZEN_WORKSPACE_ID}/query'
    # AZLOG_ZEN_ENDPOINT: str = 'http://127.0.0.1:5000/query'
    # TOKEN_URL: str = 'http://127.0.0.1:5000/token'

    if Path(AZDIAG_JSON_FILEPATH_STR).is_file():
        new_data_choice = input('Get data?')
        if new_data_choice == '':
            new_data_choice = 'n'
            logger.info(f'Choice: {new_data_choice} - No new data will be fetched.')
        else:
            logger.info(f'Choice: {new_data_choice} - New data will be fetched.')
            Path(AZDIAG_JSON_FILEPATH_STR).unlink(missing_ok=True)
    if Path(TOKEN_FILEPATH_STR).is_file():
        Path(TOKEN_FILEPATH_STR).unlink(missing_ok=True)

    FileHandler.remove_files_from_folder(folder=OUTPUT_FOLDER, file_extension=DELALL_FEXT)

    if Path(AZDIAG_JSON_FILEPATH_STR).is_file():
        query_name_for_local = UserInputHandler.select_query()
        DataProcessor.process_data(AZDIAG_JSON_FILEPATH_STR, AZDIAG_CSV_FILEPATH_STR, AZDIAG_EXTRACTIONLOGS_FILEPATH_STR, AZDIAG_EXCEL_FILEPATH, AZDIAG_EXCEL_FINAL_FILEPATH,
                                   CL_INIT_ORDER, CL_DROPPED, AZDIAG_REGEX, AZDIAG_STRINGS, MATCH_VALUE_TO_SPECIFIC_COLUMN, EXCLUSION_PAIRS, CL_FINAL_ORDER)
        logger.debug(f'Processed local data from source: {AZDIAG_JSON_FILEPATH_STR}')
        BackupManager.manage_backups(OUTPUT_FOLDER, query_name_for_local)
    else:
        data, query_name = APIManager.fetch_and_save_api_data(TOKEN_URL, CLIENT_ID, CLIENT_SECRET, AZDIAG_SCOPE, TOKEN_FILEPATH_STR, AZDIAG_JSON_FILEPATH_STR, AZLOG_ZEN_ENDPOINT)
        logger.info(f'Query name: {query_name}')
        logger.info(f'API data saved to: {AZDIAG_JSON_FILEPATH_STR}')
        if data:
            DataProcessor.process_data(AZDIAG_JSON_FILEPATH_STR, AZDIAG_CSV_FILEPATH_STR, AZDIAG_EXTRACTIONLOGS_FILEPATH_STR, AZDIAG_EXCEL_FILEPATH, AZDIAG_EXCEL_FINAL_FILEPATH,
                                       CL_INIT_ORDER, CL_DROPPED, AZDIAG_REGEX, AZDIAG_STRINGS, MATCH_VALUE_TO_SPECIFIC_COLUMN, EXCLUSION_PAIRS, CL_FINAL_ORDER)
            logger.info(f'Processed API data from source: {AZDIAG_JSON_FILEPATH_STR}')
            BackupManager.manage_backups(OUTPUT_FOLDER, query_name)
        else:
            logger.error('Failed to process API data.')


if __name__ == '__main__':
    main()
