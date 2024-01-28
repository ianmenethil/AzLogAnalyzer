import json
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
import httpx
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
# from rich.prompt import Prompt
import yaml

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            level=logging.INFO,
            console=Console(),
            show_time=True,
            omit_repeated_times=True,
            show_level=True,
            show_path=True,
            enable_link_path=True,
            highlighter=None,
            markup=True,
            rich_tracebacks=True,
            tracebacks_width=100,
            tracebacks_extra_lines=0,
            tracebacks_theme='monokai',
            tracebacks_word_wrap=True,
            tracebacks_show_locals=True,
            tracebacks_suppress=(),
            # locals_max_length=10,
            # locals_max_string=80,
            log_time_format="[%x %X]",
            keywords=["time", "name", "level", "message"])
    ])

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)
httpx_logger.addHandler(logging.StreamHandler())

# Set up logging
logger: logging.Logger = logging.getLogger(name=__name__)
logger.setLevel(level=logging.ERROR)
handler = RichHandler()
logger.addHandler(hdlr=handler)

# Read the config.yaml file
with open(file='config.yaml', mode='r', encoding='utf-8') as file:
    config: Dict[str, Any] = yaml.safe_load(stream=file)

# Constants from config
TENANT_ID: str = config['tenant_id']
CLIENT_ID: str = config['client_id']
CLIENT_SECRET: str = config['client_secret']
SUBSCRIPTION_ID: str = config['zenpay_subid']
RESOURCE: str = config['resource']
TOKEN_FILE: str = config['token_file']
JSON_FILE: str = config['json_file']
CSV_FILE: str = config['csv_file']
ZENCSV_FILE = 'zenpay' + CSV_FILE
ZENJSON_FILE = 'zenpay' + JSON_FILE
UGCSV_FILE = 'ugc' + CSV_FILE
UGCJSON_FILE = 'ugc' + JSON_FILE
OAUTH2_ENDPOINT: str = f'https://login.microsoftonline.com/{TENANT_ID}/oauth2/token'
REGULATORY_COMPLIANCE_ENDPOINT: str = f'https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}/providers/Microsoft.Security/regulatoryComplianceStandards/PCI-DSS-4/regulatoryComplianceControls?api-version=2019-01-01-preview'  # pylint: disable=line-too-long
AZLOG_ENDPOINT: str = "https://api.loganalytics.azure.com/v1/workspaces/"

console = Console()


def get_token():
    try:
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'scope': 'https://management.azure.com/.default'  # Update the scope if necessary
        }
        token_url = f'https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token'
        response = httpx.post(token_url, headers=headers, data=payload)
        response.raise_for_status()
        token_response = response.json()
        expires_in_seconds = int(token_response['expires_in'])
        return token_response['access_token'], datetime.now() + timedelta(seconds=expires_in_seconds)
    except Exception as e:
        logger.error(msg=f"An error occurred: {str(object=e)}")
    return None, None


def load_or_refresh_token() -> Any:
    try:
        with open(file=TOKEN_FILE, mode='r', encoding='utf-8') as file:
            token_info = json.load(fp=file)
            expires_at = datetime.fromisoformat(token_info['expires_at'])
            if datetime.now() >= expires_at:
                raise ValueError("Token expired")
            console.print("Token is valid and loaded from file.", style="green")
            return token_info['token']
    except FileNotFoundError:
        console.print(f"Token file '{TOKEN_FILE}' not found. Generating a new token...", style="yellow")
    except ValueError:
        console.print("Token expired. Refreshing token...", style="yellow")

    access_token, expires_at = get_token()
    with open(file=TOKEN_FILE, mode='w', encoding='utf-8') as file:
        json.dump(obj={'token': access_token, 'expires_at': expires_at.isoformat()}, fp=file)
    console.print("New token generated and saved to file.", style="green")
    return access_token


def getRegulatoryCompliance(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    try:
        response = httpx.get(url=url, headers=headers)
        response.raise_for_status()
        logger.debug(f"Request URL: {response.request.url}")
        logger.debug(f"Request Headers: {response.request.headers}")
        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Headers: {response.headers}")
        logger.debug(f"Response Content: {response.text}")
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return {}


def save_json(response_json, filename) -> None:
    with open(file=filename, mode='w', encoding='utf-8') as file:
        json.dump(obj=response_json, fp=file)


def save_csv(response_json, filename) -> None:
    df = pd.json_normalize(data=response_json, record_path='value')
    df.to_csv(filename, index=False)


def get_az_logs(query, headers):
    try:
        response = httpx.get(url=AZLOG_ENDPOINT + query, headers=headers)
        response.raise_for_status()
        logger.debug(f"Request URL: {response.request.url}")
        logger.debug(f"Request Headers: {response.request.headers}")
        logger.debug(f"Response Status Code: {response.status_code}")
        logger.debug(f"Response Headers: {response.headers}")
        logger.debug(f"Response Content: {response.text}")
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return {}


def main() -> None:
    access_token = load_or_refresh_token()
    if not access_token:
        console.print("No access token found.", style="red")
        return
    console.print(f"Access token: {access_token}", style="green")  # Debug print
    headers = {'Authorization': f'Bearer {access_token}'}
    if not headers:
        console.print("No headers found.", style="red")
        return
    console.print(f"Headers: {headers}", style="green")  # Debug print
    try:
        # CIP1 = "14.161.17.210"
        # CIP2 = '115.78.224.106'
        query = """
        {
    "query": "AzureDiagnostics | extend CIPs = coalesce(client_ip_s, clientIp_s, clientIP_s) | extend Host = coalesce(host_s, originalHost_s, host_name_s, hostname_s), Query = coalesce(originalRequestUriWithArgs_s, requestQuery_s, requestUri_s), Agent = coalesce(userAgent_s, userAgent_g), Method = httpMethod_s | where CIPs == '14.161.17.210' or CIPs == '115.78.224.106' | where TimeGenerated >= ago(2d)"
}"""
        get_az_logs(query=query, headers=headers)
    except Exception as e:
        console.print(f"An error occurred: {e}", style="red")
    # try:
    #     # response_json = get_data(url=REGULATORY_COMPLIANCE_ENDPOINT, headers=headers)
    #     if response_json:
    #         save_json(response_json=response_json, filename=JSON_FILE)
    #         save_csv(response_json=response_json, filename=CSV_FILE)
    #         console.print("Response saved as JSON and CSV.", style="green")
    # except Exception as e:
    #     console.print(f"An error occurred: {e}", style="red")


# def main() -> None:
#     access_token = load_or_refresh_token()
#     if not access_token:
#         console.print("No access token found.", style="red")
#         return
#     console.print(f"Access token: {access_token}", style="green")  # Debug print
#     headers = {'Authorization': f'Bearer {access_token}'}
#     if not headers:
#         console.print("No headers found.", style="red")
#         return
#     console.print(f"Headers: {headers}", style="green")  # Debug print
#     try:
#         response_json = get_data(url=REGULATORY_COMPLIANCE_ENDPOINT, headers=headers)
#         if response_json:
#             save_json(response_json=response_json, filename=JSON_FILE)
#             save_csv(response_json=response_json, filename=CSV_FILE)
#             console.print("Response saved as JSON and CSV.", style="green")
#     except Exception as e:
#         console.print(f"An error occurred: {e}", style="red")

if __name__ == "__main__":
    main()
