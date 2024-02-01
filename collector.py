# collector.py  # pylint disable=too-many-lines # pylint: disable=C0302
from configurator import setup_logging, console, ExcelConfigurations, RegexConfigurations, AppConfigurations, LF
import json  # pylint: disable=C0411
from typing import Any, Tuple, List, Dict, cast  # pylint: disable=C0411
import logging  # pylint: disable=C0411
import sys  # pylint: disable=C0411
from pathlib import Path  # pylint: disable=C0411
import pandas as pd  # pylint: disable=C0411
import httpx  # pylint: disable=C0411
from compareHeaders import compare_and_write_output
from timeManager import TimeManager
from configLoader import ConfigLoader
from userInputManager import UserInputHandler
from excelManager import ExcelManager
from backupManager import BackupManager
from fileManager import FileManager
from regexManager import RegexManager
from logManager import LogManager

setup_logging()
logger = logging.getLogger(__name__)
get_current_time_info = TimeManager.get_current_time_info
convert_syd_to_aztime = TimeManager.convert_syd_to_aztime
validate_starttime_endtime = TimeManager.validate_starttime_endtime
convert_utc_to_sydney = TimeManager.convert_utc_to_sydney
validate_timeago_variable = TimeManager.validate_timeago_variable
input_datetime_with_validation = TimeManager.input_datetime_with_validation

OUTPUT_FOLDER: str = 'AZLOGS'
LOG_FOLDER: str = f'{OUTPUT_FOLDER}/LOG'
# if not Path(OUTPUT_FOLDER).is_dir():
#     Path(OUTPUT_FOLDER).mkdir()
# if not Path(LOG_FOLDER).is_dir():
#     Path(LOG_FOLDER).mkdir()
CL_TimeGenerated: str = 'TimeGenerated'
CL_LocalTime: str = 'LocalTime'
DELALL_FEXT: Tuple[str, str, str, str] = ('.csv', '.xlsx', '.log', '.txt')


class APIManager():

    @staticmethod
    def get_logs_from_azure_analytics(query, headers, zen_endpoint) -> Any | dict:  # ! API call to get logs
        """The `get_logs_from_azure_analytics` function makes an API call to retrieve logs from Azure Analytics
        and returns the response as a JSON object."""
        try:
            json.dumps(headers)
            headers['Content-Type'] = 'application/json'
            response = httpx.post(url=zen_endpoint, data=query, headers=headers, timeout=60)
            response.raise_for_status()
            logger.info(f"Response Status Code: {response.status_code}")
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
            FileManager.save_json(token_response, token_file)
            return token_response
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
        except Exception as e:
            logger.error(f"An error occurred: {str(object=e)}")
        return None, None

    @staticmethod
    def fetch_and_save_api_data(token_url: str = '',
                                client_id: str = '',
                                client_secret: str = '',
                                resource_scope: str = '',
                                token_file_path: str = '',
                                json_file_path: str = '',
                                endpoint: str = ''):
        """The function fetches and saves API data using a token and query, and handles errors."""
        query_name, query_content = UserInputHandler.get_query_input()
        KQL = json.dumps({"query": query_content})
        logger.info(f'Active [yellow]Query Name:[/yellow] [red]{query_name}[/red]')
        LF()
        console.print('([red]Formatted[/red] Query):', style="blue blink", justify="left")
        LF()
        console.print(query_content, style="white", justify="left")
        LF()
        console.print('IS THE QUERY CORRECT?', style="green bold", justify="left")
        LF()
        correct_query = input(f'Query name: {query_name} Enter to continue, or type anything to exit...')
        LF()
        if correct_query == '':
            logger.info(f'Proceed with: {query_name}')
        else:
            logger.info('Incorrect query, exiting...')
            sys.exit()

        token_info = APIManager.save_new_token(token_url, client_id, client_secret, resource_scope, token_file_path)
        if not token_info:
            logger.warning("Failed to refresh access token.")
            return 'Error in token', query_name
        try:
            token_key = token_info["access_token"]  # type: ignore
            headers = {'Authorization': f'Bearer {token_key}'}
            # logger.info(f"Headers: {headers}") # !
            query = KQL
            if response := APIManager.get_logs_from_azure_analytics(query, headers, endpoint):
                FileManager.save_json(response, json_file_path)
                return response, query_name
            logger.error("Could not fetch data from API. Exiting.")
            sys.exit()
        except Exception as e:
            logger.error(f"e in fetch_and_save_api_data: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
            logger.error("Could not fetch data from API. Exiting.")
            sys.exit()


class FileHandler():

    @staticmethod
    def add_DF_to_CSVFILE_dropNA(json_data, filename: str, log_file: str):
        """Converts tables in JSON response into pandas DataFrames, drops any blank columns, and saves the resulting DataFrames as CSV files."""
        try:
            if 'tables' in json_data:
                for table in json_data['tables']:
                    df = pd.DataFrame(table['rows'], columns=[col['name'] for col in table['columns']])
                    dataframe_row_length = len(df.index)
                    dataframe_column_length = len(df.columns)
                    dataframe_column_names_before_drop = set(df.columns)
                    logger.info(f'Before dropna - DF Rows: {dataframe_row_length}, DF Columns: {dataframe_column_length}')
                    LogManager.save_removed_rows_to_raw_logs(df, log_file)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write('Before dropna - Column name,Number of rows\n')
                        for c in df.columns:
                            f.write(f'{c},{dataframe_row_length}\n')

                    # Dropping columns with all NaN values
                    df.dropna(axis=1, how='all', inplace=True)
                    dataframe_column_names_after_drop = set(df.columns)

                    # Logging and writing after dropna
                    logger.info('###After dropna###')
                    logger.info(f'After dropna - DF Rows: {len(df.index)}, DF Columns: {len(df.columns)}')

                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write('\nAfter dropna - Column name,Number of rows\n')
                        for c in df.columns:
                            f.write(f'{c},{len(df.index)}\n')

                    # Identify dropped columns due to NaN and log them
                    columns_dropped = dataframe_column_names_before_drop - dataframe_column_names_after_drop
                    if columns_dropped:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write('\nDropped columns due to all NaN\n')
                            for c in columns_dropped:
                                f.write(f'{c}\n')

                    # Saving the DataFrame to CSV after dropping NaN columns
                    df.to_csv(filename, index=False, encoding='utf-8')
                    logger.info(f'Saved {filename}')
                    return True
            else:
                logger.warning("No tables found in response.")
                return None
        except KeyError as e:
            logger.error(f"Key error in processing JSON data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    @staticmethod
    def add_LocalTime_col(input_file, source_col, dest_col, output_file) -> None:
        try:
            df = pd.read_csv(input_file, low_memory=False)
            logger.info(f"source_col: {source_col} | dest_col: {dest_col}")
            new_column_data = df[source_col].apply(convert_utc_to_sydney)
            if source_col not in df.columns:
                logger.warning(f"Column {source_col} not found in DF.")
            # ? Add LocalTime column
            else:
                df.insert(0, dest_col, new_column_data)
            df.to_csv(output_file, index=False)
            logger.info(f"DataFrame saved to CSV with new order: {output_file}")
        except Exception as e:
            logger.error(f'Error in processing data: {e}', exc_info=True, stack_info=True)


# class DataFrameManipulator():

# @staticmethod
# def processExclusionPairs(df: pd.DataFrame, filename: str, exclusion_pairs, log_file: str) -> Any | DataFrame:
#     try:
#         for column, values_to_exclude in exclusion_pairs.items():
#             if column in df.columns:
#                 logger.info(f"Column: {column} found in DF. Values to exclude: {values_to_exclude}")
#                 filter_condition = df[column].isin(values_to_exclude)
#                 # ! Capture the rows that will be removed
#                 removed_rows = df[filter_condition].copy()  # ? Use .copy() to avoid SettingWithCopyWarning
#                 # ? Log the removed rows if there are any
#                 if not removed_rows.empty:
#                     pattern = f"Exclusion: {values_to_exclude}"
#                     LogManager.save_removed_rows_to_raw_logs(removed_rows, log_file)
#                     logger.info(f"Removed rows based on column '{column}' and pattern '{pattern}' have been logged to {log_file}")
#                 # ? update df to exclude the rows
#                 df = df[~filter_condition]
#                 if df.empty:
#                     logger.info("DataFrame is empty after exclusion.")
#                     break  # ! Exit the loop if df is empty
#                 # ! Identify columns that may be dropped due to becoming entirely NaN
#                 columns_before = set(df.columns)
#                 df.dropna(axis=1, how='all')
#                 # df.dropna(axis=1, how='all', inplace=True)
#                 columns_after = set(df.columns)
#                 # columns_dropped = columns_before - columns_after  # ? NOTE
#                 # if columns_dropped:
#                 if columns_dropped := columns_before - columns_after:
#                     logger.info(f"Columns dropped because they became empty after exclusion: {columns_dropped}")
#                 else:
#                     logger.warning("No columns were dropped after exclusion.")
#         # ! After processing, save the potentially modified df
#         if not df.empty:
#             df.to_csv(filename, index=False)
#             logger.info(f"Modified DF saved to {filename}")
#         else:
#             logger.warning("DataFrame is empty after exclusions. Skipping saving to CSV.")
#         return df
#     except Exception as e:
#         logger.error(f"Error in exclusion processing: {e}", exc_info=True)
#         return df

# @staticmethod
# def find_duplicate_columns(df) -> Any:
#     """The function `find_duplicate_columns` takes a DF as input and returns a list of duplicate
#     column names."""
#     duplicate_columns = df.columns[df.columns.duplicated()]
#     # logger.info(f"Duplicate columns: {duplicate_columns.tolist()}")
#     return duplicate_columns.tolist()


class DataProcessor():

    @staticmethod
    def process_data(JSON_FILE, CSV_FILE, LOGS_FILE, EXCEL_FILE):
        try:
            JSONDATA = FileManager.read_json(filename=JSON_FILE)
            if 'tables' in JSONDATA:
                saveData = FileHandler.add_DF_to_CSVFILE_dropNA(json_data=JSONDATA, filename=CSV_FILE, log_file=LOGS_FILE)
                if saveData:
                    logger.info(f'CSV file created: {CSV_FILE}')
                else:
                    logger.error(f'CSV file not created: {CSV_FILE}. Exiting...')
                    sys.exit()
                if saveData:
                    FileHandler.add_LocalTime_col(
                        input_file=CSV_FILE,
                        source_col=CL_TimeGenerated,
                        dest_col=CL_LocalTime,
                        output_file=CSV_FILE,
                    )
                    logger.info(f'LT column added to {CSV_FILE}')
                    ExcelManager.excelCreate(input_csv_file=CSV_FILE, output_excel_file=EXCEL_FILE)
                    logger.info(f'excelCreate: Input {CSV_FILE} | Output {EXCEL_FILE}')
                    return EXCEL_FILE
        except Exception as e:
            logger.error(f'E in process_local_data: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)


# ! TIME
time_info = get_current_time_info()
NOWINSYDNEY: str = time_info['NOWINSYDNEY']
NOWINAZURE: str = time_info['NOWINAZURE']
TODAY: str = time_info['TODAY']
NOWINSYDNEY_FILEFORMAT: str = time_info['NOWINSYDNEY_FILEFORMAT']
NOWINAZURE_FILEFORMAT: str = time_info['NOWINAZURE_FILEFORMAT']
logger.info(f'Sydney Time {NOWINSYDNEY}')
logger.info(f'NOWINAZURE Time {NOWINAZURE}')

# # ? IPs
IPs_FILEPATH: str = 'config/IPs.yaml'
whitelist_data = FileManager.read_yaml(IPs_FILEPATH)
RAW_WHITELIST_IPs: List[str] = whitelist_data['WHITELIST']
WHITELIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_WHITELIST_IPs)

RAW_BLACKLIST_IPs: List[str] = whitelist_data['BLACKLIST']
BLACKLIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_BLACKLIST_IPs)

logger.info(f"WHITELIST: {WHITELIST_IPs}")
logger.info(f"BLACKLIST: {BLACKLIST_IPs}")
# ! OPTIONS
azdiag1 = 'AZDIAG_IP1IP2_TIMEAGO'
azdiag2 = 'AZDIAG_TIMEBETWEEN'
azdiag3 = 'AZDIAG_IP_TIMEBETWEEN'
azdiag4 = 'AZDIAG_TIMEAGO'
req1 = 'APPREQ_TIMEAGO'
page1 = 'APPPAGE_TIMEAGO'
browser1 = 'APPBROWSER_TIMEAGO'
httplogs1 = 'HTTPLogs_TIMEAGO'
httplogs2 = 'HTTPLogs_TIMEBETWEEN'
ipsec1 = 'APPSERVIPSec_TIMEAGO'


def main() -> None:
    """The main function loads configurations, processes data from either a local file or an API, and
    manages backups."""
    excel_config, regex_config, az_config = ConfigLoader.load_configurations()
    excel_config = cast(ExcelConfigurations, excel_config)
    regex_config = cast(RegexConfigurations, regex_config)
    az_config = cast(AppConfigurations, az_config)

    # # ! Columns
    CL_DROPPED: List[str] = excel_config['CL_DROPPED']
    CL_FINAL_ORDER: List[str] = excel_config['CL_FINAL_ORDER']
    MATCH_VALUE_TO_SPECIFIC_COLUMN: Dict[str, List[str]] = regex_config['MATCH_VALUE_TO_SPECIFIC_COLUMN']

    # # ! Configs
    TENANT_ID: str = az_config['TENANT_ID']
    CLIENT_ID: str = az_config['CLIENT_ID']
    CLIENT_SECRET: str = az_config['CLIENT_SECRET']
    ZEN_WORKSPACE_ID: str = az_config['ZEN_WORKSPACE_ID']
    AZDIAG_SCOPE: str = az_config['AZDIAG_SCOPE']
    AZLOG_ENDPOINT: str = az_config['AZLOG_ENDPOINT']
    TOKEN_FILEPATH_STR: str = az_config['TOKEN_FILEPATH_STR']
    JSON_FILE: str = az_config['AZDIAG_JSON_FILEPATH_STR']
    CSV_FILE: str = az_config['AZDIAG_CSV_FILEPATH_STR']
    EXCEL_FILE: Path = Path(az_config['AZDIAG_EXCEL_FILEPATH'])
    EXCEL_FINAL_FILE: Path = Path(az_config['AZDIAG_EXCEL_FINAL_FILEPATH'])
    LOG_FILE: str = az_config['Extraction_LogFILE']
    # TOKEN_URL: str = f'https://login.microsoftonline.com/{TENANT_ID}/oauth2/token'
    #  AZLOG_ZEN_ENDPOINT: str = f'{AZLOG_ENDPOINT}{ZEN_WORKSPACE_ID}/query'
    AZLOG_ZEN_ENDPOINT: str = 'http://127.0.0.1:5000/query'
    TOKEN_URL: str = 'http://127.0.0.1:5000/token'
    if Path(JSON_FILE).is_file():
        new_data_choice = input('Get data?')
        if new_data_choice == '':
            new_data_choice = 'n'
            logger.info(f'Choice: {new_data_choice} - No new data will be fetched.')
        else:
            logger.info(f'Choice: {new_data_choice} - New data will be fetched.')
            Path(JSON_FILE).unlink(missing_ok=True)
    if Path(TOKEN_FILEPATH_STR).is_file():
        Path(TOKEN_FILEPATH_STR).unlink(missing_ok=True)
    FileManager.remove_files_from_folder(folder=OUTPUT_FOLDER, file_extension=DELALL_FEXT)
    LF()
    if Path(JSON_FILE).is_file():
        # query_name_for_local = UserInputHandler.select_query()
        DataProcessor.process_data(JSON_FILE, CSV_FILE, LOG_FILE, EXCEL_FILE)
        logger.info(f'Processed local data from source: {JSON_FILE}')
        logger.info(f'Excel file saved to: {EXCEL_FILE}')
        # BackupManager.manage_backups(OUTPUT_FOLDER, query_name_for_local)

    else:
        data, query_name = APIManager.fetch_and_save_api_data(TOKEN_URL, CLIENT_ID, CLIENT_SECRET, AZDIAG_SCOPE, TOKEN_FILEPATH_STR, JSON_FILE, AZLOG_ZEN_ENDPOINT)
        logger.info(f'Query name: {query_name}')
        logger.info(f'API data saved to: {JSON_FILE}')
        if data:
            DataProcessor.process_data(JSON_FILE, CSV_FILE, LOG_FILE, EXCEL_FILE)
            logger.info(f'Processed API data from source: {JSON_FILE}')
            logger.info(f'Excel file saved to: {EXCEL_FILE}')
            # BackupManager.manage_backups(OUTPUT_FOLDER, query_name)
        else:
            logger.error('Failed to process API data.')

    # At this point Excel should be ready after the refactor we need to apply regex from here onwards
    data = pd.read_excel(EXCEL_FILE)
    try:
        RegexManager.filter_df_based_on_patterns(df=data, config_dict=MATCH_VALUE_TO_SPECIFIC_COLUMN, output_dir=OUTPUT_FOLDER)
    except Exception as e:
        logger.error(f'Error in filter_df_based_on_patterns: {e}', exc_info=True, stack_info=True)

    try:
        EXCEL_FILE_STR = str(EXCEL_FILE)
        EXCEL_FINAL_FILE_STR = str(EXCEL_FINAL_FILE)
        ExcelManager.create_final_excel(input_file=EXCEL_FILE_STR,
                                        output_file=EXCEL_FINAL_FILE_STR,
                                        logfile=LOG_FILE,
                                        columns_to_be_dropped=CL_DROPPED,
                                        final_column_order=CL_FINAL_ORDER)
        logger.info(f'Excel file saved to: {EXCEL_FINAL_FILE}')
    except Exception as e:
        logger.error(f'Error in create_final_excel: {e}', exc_info=True, stack_info=True)
    #     remove_patterns(dataframe, extraction_file, regex, string, key_col_to_val_patts) -> Any:
    #  remove_regex_patterns(df, regex, string, columns_to_search, extraction_file) -> Any:
    # remove_key_value_patterns(df, key_col_to_val_patts, extraction_file):
    # def processExclusionPairs(df: pd.DataFrame, filename: str, exclusion_pairs, log_file: str) -> Any | DataFrame:
    # then create final excel
    LF()
    COMPARE_HEADER_FILE = 'AZLOGS/COMPARE_HEADER.xlsx'
    compare_and_write_output(CSV_FILE, EXCEL_FINAL_FILE, COMPARE_HEADER_FILE)


if __name__ == '__main__':
    main()
