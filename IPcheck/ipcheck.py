import logging
import json
import os
from typing import Dict, Optional, Any, Literal, Union
import yaml
import httpx
import pandas as pd
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
import click

logging.basicConfig(filename='api_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


def save_response_to_json(data: Dict[str, Any], identifier: str, endpoint: str) -> str:
    """Save the API response data to a unique JSON file and return the file path."""
    filename = f"./{identifier}_{endpoint}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved API response to {filename}")
    return filename


def print_json_data(json_data: Dict[str, Any]) -> None:
    """Print JSON data in colorful table format."""
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    for key, value in json_data.items():
        table.add_column(key)
        table.add_row(str(value))
    console.print(table)


def load_config(filename: str) -> Optional[Dict[str, Any]]:
    """Load configuration from a YAML file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {filename}")
            return cfg
    except yaml.YAMLError as exc:
        logger.error(f"Error loading configuration from {filename}: {exc}")
        return None


def make_api_call(url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Make an API call with user approval of request details."""
    try:
        response = httpx.get(url, headers=headers)
        logger.info(f"Made request to {url}. status code: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        logger.error(f"Request to {url} failed with status code {response.status_code}")
        return None
    except Exception as exc:
        logger.error(f"Exception during request to {url}: {exc}")
        return None


def read_json(filename: str) -> Optional[Dict[str, Any]]:
    """Read JSON data from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Read data from {filename}")
            return data
    except Exception as exc:
        logger.error(f"Error reading data from {filename}: {exc}")
        return None


def append_data_to_df_and_save_from_json(json_file: str, output_file: str, sheet_name: str, ip: str) -> None:
    """Append data from a JSON file to a DataFrame and save it to an Excel file."""
    if not (json_data := read_json(json_file)):
        return
    df = pd.DataFrame([{'IP Address': ip, **json_data}])

    # Check if the Excel file exists
    # mode = 'w' if not os.path.exists(output_file) else 'a'
    mode: Union[Literal['w'], Literal['a']] = 'a' if os.path.exists(output_file) else 'w'
    with pd.ExcelWriter(output_file, mode=mode, engine='openpyxl', if_sheet_exists='overlay' if mode == 'a' else None) as writer:
        try:
            if mode == 'a':
                existing_df = pd.read_excel(output_file, sheet_name=sheet_name)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                logger.info(f"Appending data from {json_file} to {output_file} in sheet '{sheet_name}'.")
            else:
                logger.info(f"Creating new sheet '{sheet_name}' in {output_file}.")
                combined_df = df
        except (FileNotFoundError, ValueError):
            logger.error(f"Failed to load existing data from {output_file} in sheet '{sheet_name}'. Starting with new data.")
            combined_df = df

        combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Data for IP {ip} appended and saved to {output_file} in sheet '{sheet_name}'.")


def get_malicious_info(IP: str, key: str) -> Any | None:
    malurl = f"https://api.criminalip.io/v1/feature/ip/malicious-info?ip={IP}"
    headers = {'x-api-key': key}
    try:
        response = httpx.get(malurl, headers=headers)
        if response.status_code == 200:
            json_data = response.json()
            logger.info(f"Successful request: {malurl} with response: {json_data}")
            return json_data
        logger.error(f"Failed request: {malurl} with status code: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Exception during request: {malurl} with error: {str(e)}")
        return None


def get_privacy_threat(IP: str, api_key: str) -> Any | None:
    pt_url = f"https://api.criminalip.io/v1/feature/ip/privacy-threat?ip={IP}"
    headers = {'x-api-key': api_key}
    try:
        response = httpx.get(pt_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successful request: {pt_url} with response: {data}")
            return data
        logger.error(f"Failed request: {pt_url} with status code: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Exception during request: {pt_url} with error: {str(e)}")
        return None


def get_suspicious_info(IP: str, api_key: str) -> Any | None:
    si_url = f"https://api.criminalip.io/v1/feature/ip/suspicious-info?ip={IP}"
    headers = {'x-api-key': api_key}
    try:
        response = httpx.get(si_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successful request: {si_url} with response: {data}")
            return data
        else:
            logger.error(f"Failed request: {si_url} with status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception during request: {si_url} with error: {str(e)}")
        return None


def call_CRIMINALIP(IP: str, api_key: str, end_point: str, api_name: str) -> Optional[Dict[str, Any]]:
    """Generalized function to fetch data from different endpoints."""
    url = f"https://api.criminalip.io/v1/feature/ip/{end_point}?ip={IP}"
    headers = {"x-api-key": api_key}
    logger.info(f"Preparing {api_name} call for IP: {IP}")
    return make_api_call(url, headers)


def call_IPQS(IP: str, api_key: str, agent: Optional[str] = None, lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Call the IP Quality Score API with the specified IP and API key, including optional parameters."""
    base_url = f"https://www.ipqualityscore.com/api/json/ip/{api_key}/{IP}"
    params = {"fast": "true", "lighter_penalties": "false", "mobile": "true", "transaction_strictness": "1", "strictness": "1", "allow_public_access_points": "true"}
    # Conditionally add optional parameters if provided
    if agent:
        params["user_agent"] = agent
    if lang:
        params["user_language"] = lang

    url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
    logger.info(f"Preparing IPQS call for IP: {IP} with params: {params}")
    return make_api_call(url, {})


# ! doesnt work atm
# def flatten_and_expand_df(df):
#     logger.info("Starting the expansion and flattening process.")
#     temp_df_list = []

#     for index, row in df.iterrows():
#         expanded = False  # Flag to track if the row has been expanded
#         for col in df.columns:
#             if isinstance(row[col], dict) and 'data' in row[col] and isinstance(row[col]['data'], list):
#                 expanded = True
#                 for item in row[col]['data']:
#                     new_row = row.copy()
#                     for key, value in item.items():
#                         new_col_name = f"{col}_{key}"
#                         new_row[new_col_name] = value
#                     temp_df_list.append(new_row)
#                 logger.info(f"Expanded '{col}' into separate columns.")
#                 break  # Break after expanding to avoid adding the original row again
#         if not expanded:
#             # If no expansion was done for this row, add it as is
#             temp_df_list.append(row)

#     if not temp_df_list:
#         logger.warning("No data was expanded. The original DataFrame might not contain the expected nested structures.")
#         return df

#     expanded_df = pd.DataFrame(temp_df_list).reset_index(drop=True)
#     expanded_df = expanded_df.drop_duplicates().reset_index(drop=True)

#     logger.info("Completed the expansion and flattening process.")
#     return expanded_df


@click.command()
@click.option('--ip', default=None, help='The IP address to query.')
@click.option('--agent', default=None, help='Optional user agent string.')
@click.option('--lang', default=None, help='Optional user language.')
def main(ip: Optional[str], agent: Optional[str], lang: Optional[str]):
    config = load_config('config.yaml')
    if not config:
        logger.error("Configuration loading failed. Exiting.")
        return

    if not ip:
        ip = input("Enter the IP address: ")
    OUTFILE = f"./{ip}.xlsx"

    if not agent:
        agent = None
    if not lang:
        lang = None

    IPQS_KEY = config.get("IPQS_KEY", "")
    IPCRIM_KEY = config.get("IPCRIMINAL_KEY", "")

    if data_ipqs := call_IPQS(ip, IPQS_KEY, agent, lang):
        IPQS_JSON_FILE = save_response_to_json(data_ipqs, ip, 'IPQS')
        append_data_to_df_and_save_from_json(IPQS_JSON_FILE, OUTFILE, 'IPQS', ip)
        append_data_to_df_and_save_from_json(IPQS_JSON_FILE, OUTFILE, 'All', ip)

    # Criminal IP calls
    for endpoint in ["malicious-info", "privacy-threat", "suspicious-info"]:
        if ipcriminal_data := call_CRIMINALIP(IP=ip, api_key=IPCRIM_KEY, end_point=endpoint, api_name=endpoint):

            IPCRIMINAL_JSON_FILE = save_response_to_json(ipcriminal_data, ip, endpoint)
            append_data_to_df_and_save_from_json(IPCRIMINAL_JSON_FILE, OUTFILE, 'ipcrim', ip)
            append_data_to_df_and_save_from_json(IPCRIMINAL_JSON_FILE, OUTFILE, 'All', ip)
    OUTFILE = r'F:\_Azure\AzLogAnalyzer\IPcheck\116.50.58.180.xlsx'

    # ! this part doesnt work atm flattened json
    # if os.path.exists(OUTFILE):
    #     df = pd.read_excel(OUTFILE, sheet_name='ipcrim')
    #     logger.info(f"Loaded data from {OUTFILE} sheet 'ipcrim'. Preparing to expand and flatten nested structures.")
    #     expanded_df = flatten_and_expand_df(df)
    #     expanded_file_path = f"{ip}_expanded.xlsx"
    #     expanded_df.to_excel(expanded_file_path, index=False)
    #     logger.info(f"Expanded data saved to {expanded_file_path}")


if __name__ == '__main__':
    main(ip=None, agent=None, lang=None)
