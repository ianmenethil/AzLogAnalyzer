import logging
import json
import os
from typing import Dict, Optional, Any, Literal, Union, List
import yaml
import httpx
import pandas as pd
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.pretty import Pretty
from rich.traceback import install
from rich import print
from rich.markup import escape
import exc
import click
import csv


class CustomRichHandler(RichHandler):

    def get_level_text(self, record) -> Text:
        log_level = record.levelname
        if log_level == "INFO":
            return Text(log_level, style="bold magenta")
        elif log_level == "WARNING":
            return Text(log_level, style="bold yellow")
        elif log_level == "ERROR":
            return Text(log_level, style="bold red")
        elif log_level == "DEBUG":
            return Text(log_level, style="bold blue")
        return super().get_level_text(record)


console = Console()
install()
keywords: list[str] = [
    "dropped",
    "bold blue",
]


def setup_logging() -> None:
    global console  # pylint: disable=global-variable-not-assigned
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s [%(funcName)s]",
        datefmt="[%X]",
        handlers=[
            CustomRichHandler(
                level=logging.INFO,
                console=console,
                show_time=True,
                omit_repeated_times=True,
                show_level=True,
                show_path=True,
                enable_link_path=True,
                markup=True,
                rich_tracebacks=True,
                # tracebacks_width=100,
                tracebacks_extra_lines=0,
                tracebacks_theme="monokai",
                tracebacks_word_wrap=True,
                tracebacks_show_locals=False,
                # tracebacks_suppress=(),
                # locals_max_length=10,
                locals_max_string=140,
                log_time_format="[%X]",
                keywords=keywords),
            logging.FileHandler('apilogs.log', mode='w', encoding='utf-8', delay=False, errors='ignore')
        ])


def jprint(data, level=logging.INFO) -> Any:
    global console  # pylint: disable=global-variable-not-assigned
    try:
        pretty = Pretty(data)
        console.print(pretty)
        if level == logging.DEBUG:
            logger = logging.getLogger(__name__)  # pylint: disable=redefined-outer-name
            logger.debug(data)
            return data
        return data
    except Exception as e:
        local_logger = logging.getLogger(__name__)
        local_logger.error(f"Error in jprint: {e}, data: {data}")
        return data


logging.basicConfig(filename='output/apilogs.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


def print_row_details_as_csv(df, file) -> bool:
    try:
        with open(file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(df.columns)
            for index, row in df.iterrows():
                writer.writerow(row)
                # logger.info("Row details have been printed as a CSV file.")
                return True
        print("Row details have been printed as a CSV file.")
    except Exception as e:
        logger.error(f"Error printing row details as a CSV file: {e}")
    return False


class FileHandler():

    @staticmethod
    def save_response_to_json(data: Dict[str, Any], identifier: str, endpoint: str) -> str:
        """Save the API response data to a unique JSON file and return the file path."""
        filename = f"./{identifier}_{endpoint}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved API response to {filename}")
        return filename

    @staticmethod
    def print_json_data(json_data: Union[Dict[str, Any], List[Any]]) -> None:
        """Print JSON data in colorful table format."""
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                table.add_column(key)
                table.add_row(str(value))
        elif isinstance(json_data, list):
            for item in json_data:
                table.add_column("Item")
                table.add_row(str(item))
        console.print(table)

    @staticmethod
    def load_config(filename: str) -> Optional[Dict[str, Any]]:
        """Load configuration from a YAML file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {filename}")
                return cfg
        except yaml.YAMLError as err:
            logger.error(f"Error loading configuration from {filename}: {err}")
            return None

    @staticmethod
    def read_json(filename: str) -> Optional[Dict[str, Any]]:
        """Read JSON data from a file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
                # logger.info(f"Read data from {filename}")
        except Exception as err:
            logger.error(f"Error reading data from {filename}: {err}")
            return None

    # @staticmethod
    # def append_data_to_df_and_save_from_json(json_file: str, output_file: str, sheet_name: str, ip: str) -> None:
    #     """Append data from a JSON file to a DataFrame and save it to an Excel file."""
    #     if not (json_data := FileHandler.read_json(json_file)):
    #         return
    #     try:
    #         df = pd.DataFrame([{**json_data}])
    #         if df.empty:
    #             logger.error(f"Empty DataFrame for JSON data from {json_file}")
    #             try:
    #                 df = pd.DataFrame(json_data)
    #                 logger.info(f'Row details for each row are {df.shape}')
    #                 print_row_details_as_csv(df, 'output/append_data_to_df_and_save_from_json.csv')
    #                 logger.error('Error converting JSON data to DataFrame:')
    #                 df = pd.DataFrame(json_data)
    #                 logger.info(f'Row details for each row are {df.shape}')
    #                 print_row_details_as_csv(df, 'output/append_data_to_df_and_save_from_json.csv')
    #             except Exception as e:
    #                 logger.error(f"Error converting JSON data to DataFrame: {e}")
    #                 return
    #         df['ip'] = ip
    #         df.to_excel(output_file, sheet_name=sheet_name, index=False)
    #         logger.info(f"Appended data from {json_file} to {output_file} in sheet '{sheet_name}'.")
    #     except ValueError as err:
    #         logger.error(f"Empty DataFrame for JSON data from {json_file}")
    #         logger.error(f"Error converting JSON data to DataFrame: {err}")
    #         df = pd.DataFrame(json_data)
    #         logger.info(f'Row details for each row are {df.shape}')
    #         print_row_details_as_csv(df, 'output/append_data_to_df_and_save_from_json.csv')
    #     except Exception as exc:
    #         logger.error(f"Error converting JSON data to DataFrame: {exc}")
    #         return

        # logger.info(f'Loaded dataframe {df}')
        logger.info(f'Row details for each row are {df.shape}')
        print_row_details_as_csv(df, 'output/append_data_to_df_and_save_from_json.csv')
        mode: Union[Literal['w'], Literal['a']] = 'w' if os.path.exists(output_file) else 'a'
        with pd.ExcelWriter(output_file, engine='openpyxl', mode=mode) as writer:
            try:
                if mode == 'a':
                    logger.info(f"Loading existing data from {output_file} in sheet '{sheet_name}'.")
                    input('halt')
                    existing_df = pd.read_excel(output_file, sheet_name=sheet_name)
                    # drop all colunmns that are blank
                    print_row_details_as_csv(existing_df, 'output/existing_df.csv')
                    logger.info(f'Existing rows and cols count in existing_df {existing_df.shape}')
                    existing_df = existing_df.dropna(axis=1, how='all')
                    logger.info(f'After dropna rows and cols count in existing_df {existing_df.shape}')
                    existing_df = pd.concat([existing_df, df], ignore_index=True)
                    logger.info(f'After Existing rows and cols count in existing_df {existing_df.shape}')
                    existing_df = existing_df.applymap(lambda x: tuple(x) if isinstance(x, dict) else x)  # type: ignore # Convert DataFrame columns to hashable type
                    existing_df.drop_duplicates(inplace=True)
                    logger.info(f'After comb Existing rows and cols count in existing_df {existing_df.shape}')
                    combined_df = existing_df
                    logger.info(f"combined_df{combined_df} in sheet '{sheet_name}'.")
                    print_row_details_as_csv(existing_df, 'output/existing_df1.csv')
                    input(';halt')
                else:
                    logger.info(f"Creating new sheet '{sheet_name}' in {output_file}.")
                    combined_df = df
                    logger.warning(f'Else Existing rows and cols count in combined {combined_df.shape}')

            except (FileNotFoundError, ValueError):
                logger.error(f"Failed to load existing data from {output_file} in sheet '{sheet_name}'. Starting with new data.")
                combined_df = df

            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Data for IP {ip} appended and saved to {output_file} in sheet '{sheet_name}'.")

    @staticmethod
    def combine_and_drop_columns(df: pd.DataFrame, column_combinations: Dict[str, List[str]], keep_columns: List[str]) -> pd.DataFrame:
        for new_column, columns_to_combine in column_combinations.items():
            df_row_count = len(df.index)
            df_col_count = len(df.columns)
            # Check if columns to combine exist in the dataframe
            existing_columns = [col for col in columns_to_combine if col in df.columns]

            # Skip combination if no columns exist
            if not existing_columns:
                continue

            # Combine existing columns into a new column
            df[new_column] = df[existing_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

            # Drop the original columns, except for those specified to keep
            columns_to_drop = [col for col in existing_columns if col not in keep_columns]
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            # drop any column that is empty
            df.dropna(how='all', axis=1, inplace=True)
            logger.info(f"Combined columns {columns_to_combine} into new column {new_column}.")
            latest_row_count = len(df.index)
            latest_col_count = len(df.columns)
            logger.info(f'Returning DF row count: {df_row_count} -> {latest_row_count}')
            logger.info(f'Latest column count: {df_col_count} -> {latest_col_count}')
        return df

    @staticmethod
    def delete_all_json_indir(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".json"):
                os.remove(os.path.join(folder, filename))
                logger.info(f"Deleted {filename} from {folder}.")
        logger.info(f"All JSON files deleted from {folder}.")
        input("Press Enter to continue...")
        return


class APIMan():

    @staticmethod
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

    @staticmethod
    def call_CRIMINALIP(IP: str, api_key: str, end_point: str, api_name: str) -> Optional[Dict[str, Any]]:
        """Generalized function to fetch data from different endpoints."""
        url = f"https://api.criminalip.io/v1/feature/ip/{end_point}?ip={IP}"
        headers = {"x-api-key": api_key}
        logger.info(f"Preparing {api_name} call for IP: {IP}")
        return APIMan.make_api_call(url, headers)

    @staticmethod
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
        return APIMan.make_api_call(url, {})

    @staticmethod
    def call_ipgeolocation(IP: str, IPGEO_KEY: str) -> Dict[str, Any] | None:
        url = f'https://api.ipgeolocation.io/ipgeo?apiKey={IPGEO_KEY}&ip={IP}'
        headers = {"Content-Type": "application/json"}  # Add your desired headers here
        logger.info(f"Preparing ipgeolocation call for IP: {IP}")
        logger.debug(f"Request headers: {headers}")
        return APIMan.make_api_call(url, headers)

    @staticmethod
    def incolumitas(IP: str) -> Dict[str, Any] | None:
        url = f'https://api.incolumitas.com/?q={IP}'
        logger.info(f"Preparing Incolumitas call for IP: {IP}")
        return APIMan.make_api_call(url, {})

    @staticmethod
    def ip_apicom(IP: str) -> Dict[str, Any] | None:
        url = f'http://ip-api.com/json/{IP}?fields=status,message,continent,continentCode,country,countryCode,region,regionName,city,district,zip,lat,lon,timezone,offset,currency,isp,org,as,asname,reverse,mobile,proxy,hosting,query'
        logger.info(f"Preparing IP API call for IP: {IP}")
        return APIMan.make_api_call(url, {})


# ip1 = input("Enter the IP address: ")
ip1 = '116.50.58.180'
OUTDIR: str = 'output'
OUTFILE: str = f"{OUTDIR}/{ip1}.xlsx"
import sys


@click.command()
@click.option('--ip', default=None, help='The IP address to query.')
@click.option('--agent', default=None, help='Optional user agent string.')
@click.option('--lang', default=None, help='Optional user language.')
def main(ip: Optional[str], agent: Optional[str], lang: Optional[str]) -> None:
    config = FileHandler.load_config('config.yaml')
    ip = ip1
    if not config:
        logger.error("Configuration loading failed. Exiting.")
        return
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    if not ip:
        ip = input("Enter the IP address: ")
        if ip == '':
            logger.error("No IP address provided. Exiting.")
            ip = input("Enter the IP address: ")

        else:
            logger.info(f"IP address: {ip}")
        OUTFILE = f"{OUTDIR}/{ip}.xlsx"
        if os.path.exists(OUTFILE):
            os.remove(OUTFILE)
            FileHandler.delete_all_json_indir(OUTDIR)
    # if os.path.exists(OUTFILE):
    #     os.remove(OUTFILE)
    #     logger.info(f"Removed existing file: {OUTFILE}")

    if not agent:
        agent = None
    if not lang:
        lang = None
    x = input('api?')
    # x = ''
    if x == '':
        logger.info(f"Skipping API calls")
    IPQS_KEY = config.get("IPQS_KEY", "")
    IPCRIM_KEY = config.get("IPCRIMINAL_KEY", "")
    IPGEO_KEY = config.get("IPGEO_KEY", "")
    logger.info(f"IPCRIMINAL_KEY: {IPCRIM_KEY}")
    logger.info(f"IPGEO_KEY: {IPGEO_KEY}")
    OUTFILE = f"{OUTDIR}/{ip}.xlsx"

    data_ipqs = APIMan.call_IPQS(ip, IPQS_KEY, agent, lang)
    if data_ipqs := APIMan.call_IPQS(ip, IPQS_KEY, agent, lang):
        IPQS_JSON_FILE = FileHandler.save_response_to_json(data_ipqs, ip, 'IPQS')
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPQS_JSON_FILE, output_file=OUTFILE, sheet_name='IPQS', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPQS_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

    geoip = APIMan.call_ipgeolocation(ip, IPGEO_KEY)
    if geoip := APIMan.call_ipgeolocation(ip, IPGEO_KEY):
        GEOIP_JSON_FILE = FileHandler.save_response_to_json(data=geoip, identifier=ip, endpoint='GEOIP')
        FileHandler.append_data_to_df_and_save_from_json(json_file=GEOIP_JSON_FILE, output_file=OUTFILE, sheet_name='GEOIP', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=GEOIP_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)
        logger.info(f"Preparing INCOLUMITAS call for IP: {ip} with params: {IPCRIM_KEY}")
    incolumitas = APIMan.incolumitas(ip)
    if incolumitas := APIMan.incolumitas(ip):
        INCOLUMITAS_JSON_FILE = FileHandler.save_response_to_json(data=incolumitas, identifier=ip, endpoint='INCOLUMITAS')
        FileHandler.append_data_to_df_and_save_from_json(json_file=INCOLUMITAS_JSON_FILE, output_file=OUTFILE, sheet_name='INCOLUMITAS', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=INCOLUMITAS_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

    ip_apicom = APIMan.ip_apicom(ip)
    if ip_apicom := APIMan.ip_apicom(ip):
        IPAPICOM_JSON_FILE = FileHandler.save_response_to_json(data=ip_apicom, identifier=ip, endpoint='IPAPICOM')
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPAPICOM_JSON_FILE, output_file=OUTFILE, sheet_name='ip_apicom', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPAPICOM_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)
    for endpoint in ["malicious-info", "privacy-threat", "suspicious-info"]:
        if ipcriminal_data := APIMan.call_CRIMINALIP(IP=ip, api_key=IPCRIM_KEY, end_point=endpoint, api_name=endpoint):
            IPCRIMINAL_JSON_FILE = FileHandler.save_response_to_json(ipcriminal_data, ip, endpoint)
            logger.info(f"IPCRIMINAL_JSON_FILE: {IPCRIMINAL_JSON_FILE} {ip} {endpoint} {ipcriminal_data}")
            FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIMINAL_JSON_FILE, output_file=OUTFILE, sheet_name='CriminialIP', ip=ip)
            FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIMINAL_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

        df = pd.read_excel(OUTFILE, sheet_name='All')

    else:
        IPAPICOM_JSON_FILE = f"{OUTDIR}/{ip}_IPAPICOM.json"
        IPCRIM_MAL = f"{OUTDIR}/{ip}_malicious-info.json"
        IPCRIM_PRIV = f"{OUTDIR}/{ip}_privacy-threat.json"
        IPCRIM_SUS = f"{OUTDIR}/{ip}_suspicious-info.json"
        IPQS_JSON_FILE = f"{OUTDIR}/{ip}_IPQS.json"
        GEOIP_JSON_FILE = f"{OUTDIR}/{ip}_IPGEO.json"
        INCOLUMITAS_JSON_FILE = f"{OUTDIR}/{ip}_INCOLUMITAS.json"
        OUTFILE = f"{OUTDIR}/{ip}.xlsx"
        if not os.path.exists(IPCRIM_MAL):
            sys.exit("IPCRIM_MAL file is missing")
        if not os.path.exists(IPCRIM_PRIV):
            sys.exit("IPCRIM_PRIV file is missing")
        if not os.path.exists(IPCRIM_SUS):
            sys.exit("IPCRIM_SUS file is missing")
        if not os.path.exists(IPQS_JSON_FILE):
            sys.exit("IPQS_JSON_FILE file is missing")
        # if not os.path.exists(GEOIP_JSON_FILE):
        # sys.exit("GEOIP_JSON_FILE file is missing")
        if not os.path.exists(INCOLUMITAS_JSON_FILE):
            sys.exit("INCOLUMITAS_JSON_FILE file is missing")
        if not os.path.exists(IPAPICOM_JSON_FILE):
            sys.exit("IPAPICOM_JSON_FILE file is missing")
        FileHandler.read_json(IPCRIM_MAL)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIM_MAL, output_file=OUTFILE, sheet_name='CriminialIP', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIM_MAL, output_file=OUTFILE, sheet_name='All', ip=ip)

        FileHandler.read_json(IPCRIM_PRIV)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIM_PRIV, output_file=OUTFILE, sheet_name='CriminialIP', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIM_PRIV, output_file=OUTFILE, sheet_name='All', ip=ip)

        FileHandler.read_json(IPCRIM_SUS)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIM_SUS, output_file=OUTFILE, sheet_name='CriminialIP', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPCRIM_SUS, output_file=OUTFILE, sheet_name='All', ip=ip)
        FileHandler.read_json(IPQS_JSON_FILE)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPQS_JSON_FILE, output_file=OUTFILE, sheet_name='IPQS', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPQS_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

        FileHandler.read_json(GEOIP_JSON_FILE)
        FileHandler.append_data_to_df_and_save_from_json(json_file=GEOIP_JSON_FILE, output_file=OUTFILE, sheet_name='GEOIP', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=GEOIP_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

        FileHandler.read_json(INCOLUMITAS_JSON_FILE)
        FileHandler.append_data_to_df_and_save_from_json(json_file=INCOLUMITAS_JSON_FILE, output_file=OUTFILE, sheet_name='INCOLUMITAS', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=INCOLUMITAS_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

        FileHandler.read_json(IPAPICOM_JSON_FILE)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPAPICOM_JSON_FILE, output_file=OUTFILE, sheet_name='IPAPICOM', ip=ip)
        FileHandler.append_data_to_df_and_save_from_json(json_file=IPAPICOM_JSON_FILE, output_file=OUTFILE, sheet_name='All', ip=ip)

        df = pd.read_excel(OUTFILE, sheet_name='All')

        # all_cols = df.columns.tolist()  # col list
        all_rows = df.shape[0]  # row count
        # dftypes = df.dtypes  # This line gets the data types of each column.
        # dfinfo = df.info  # This line gets a concise summary of the DataFrame, including the number of non-null entries in each column.
        # logger.info(f'dtypes {dftypes}\ndfinfo {dfinfo}')
        logger.info(f'Rows: {all_rows}')
        # logger.info(f'Columns: {all_cols}')
        # dfmemory = df.memory_usage  #A This line gets the memory usage of each column.
        # dfdescription = df.describe  # This line gets the descriptive statistics of the DataFrame, such as mean, median, etc.
        # dfisna = df.isna  # These lines get a Boolean mask where True indicates missing or NA values.
        # dfisnull = df.isnull  # These lines get a Boolean mask where True indicates missing or NA values.
        # dfnotna = df.notna  # These lines get a Boolean mask where True indicates non-missing values.
        # dfnotnull = df.notnull  # These lines get a Boolean mask where True indicates non-missing values.
        # dfnunique = df.nunique  # This line gets the number of unique values in each column.
        # dfsum = df.sum  # This line gets the sum of values for each column.
        # dfmin = df.min  # These lines get the minimum and maximum values in each column, respectively.
        # dfmax = df.max  # These lines get the minimum and maximum values in each column, respectively.
        # dfmean = df.mean  # These lines get the mean and median of each column, respectively.
        # dfmedian = df.median  # These lines get the mean and median of each column, respectively.
        # dfmode = df.mode  # This line gets the mode(s) of each column.
        # dfstd = df.values.std  # This line gets the standard deviation of the DataFrame values.

        columns_to_drop = ['elapsed_ms', 'request_id']  # Fix: Close the square bracket and remove the extra comma

        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        if df.empty:
            raise ValueError("The DataFrame is empty.")
        logger.info(f'Row count: {df.shape[0]}')

        # FileHandler.print_json_data(df.to_dict(orient='records'))  # type: ignore
        # logger.info(f"Data for IP {ip} printed in table format.")
        jprint(df.to_dict(orient='records'), level=logging.DEBUG)
        # logger.info(f"Data for IP {ip} printed in JSON format.")
        # logger.info(f"Data for IP {ip} printed in table format.")

        is_abuser = ['score', 'fraud_score', 'is_abuser', 'recent_abuse', 'abuse_record_count', 'abuse_velocity', 'active_tor', 'tor', 'is_tor']
        mal_comb = ['malicious_info', 'malicious_info']
        isvpn = ['isvpn', 'isvpn', 'active_vpn', 'vpn', 'proxy', 'is_proxy']
        dvice = ['webcam', 'iot', 'nas', 'other_ports']
        noidea = ['mobile', 'is_mobile', 'is_bogon']
        what = ['connection_type', 'hosting', 'host']
        isp = ['as', 'ISPm', 'isp', 'asname', 'ASN', 'organization', 'org']
        geo = ['lon', 'lat', 'longitude', 'latitude']
        loc = ['country', 'city', 'countryCode', 'country_code', 'region', 'regionName', 'district', 'zip', 'zip_code', 'continentCode', 'continent', 'timezone', 'offset']
        saywhat = ['ip_category', 'reverse', 'representative_domain', 'admin_page']
        port = ['can_remote_access', 'remote_port', 'current_opened_port[]']
        vuln = ['ids vulnerability', 'issues', 'scanning_record']
        bots = ['bot_status', 'is_crawler']
        is_datacenter = ''
        rir = ''

        column_combinations = {
            'abuse': is_abuser,
            'malicious': mal_comb,
            'vpn': isvpn,
            'device': dvice,
            'noidea': noidea,
            'what': what,
            'isp': isp,
            'geo': geo,
            'loc': loc,
            'saywhat': saywhat,
            'port': port,
            'vuln': vuln,
            'bot': bots,
            'is_datacenter': is_datacenter,
            'rir': rir
        }
        # save all existiing columns to a list before combination, save row count to another variable so that we can check after combination and see the difference
        existing_columns = df.columns.tolist()
        row_count = df.shape[0]
        # Combine the columns
        try:
            for key, value in column_combinations.items():
                if missing_columns := [col for col in value if col not in existing_columns]:
                    logger.info(f"Columns {missing_columns} are missing. Skipping combination for {key}.")
                    continue
                df[key] = df[value].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                df.drop(value, axis=1, inplace=True)
            # save all existing columns to a list after combination
            existing_columns_after_combination = df.columns.tolist()
            # Check if the number of rows has changed after combination
            if row_count != df.shape[0]:
                logger.info(f"Number of rows has changed after combination. "
                            f"Number of rows before combination: {row_count}. "
                            f"Number of rows after combination: {df.shape[0]}.")
            # Check if the number of columns has changed after combination
            if len(existing_columns_after_combination) != len(df.columns):
                logger.info("Number of columns has changed after combination.")
            else:
                logger.info("Number of columns has not changed after combination.")
            # Check if the columns have changed after combination
            if set(existing_columns_after_combination) != set(existing_columns):
                logger.info("Columns have changed after combination.")
            else:
                logger.info("Columns have not changed after combination.")
            # Save the combined data to a new file
            df.to_excel(OUTFILE, index=False)
            logger.info(f"Combined data saved to {OUTFILE}.")
            # Print the combined data
            jprint(df.to_dict(orient='records'), level=logging.DEBUG)
            logger.info(f"Data for IP {ip} printed in JSON format.")
            logger.info(f"Data for IP {ip} printed in table format.")
        except Exception as e:
            logger.error(f"Failed to combine data for IP {ip}: {e}")


if __name__ == '__main__':
    main()
