# configurator.py
import os
import json
import logging
from pathlib import Path
from typing import Any, List, Dict, TypedDict
import re
import yaml
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler
from rich.pretty import Pretty
from rich.text import Text

COLUMN_CONFIG_FILE = 'config/column_config.yaml'
CONFIG_FILE = 'config/config.yaml'
OUTPUT_FOLDER: str = 'AZLOGS'
LOG_FOLDER: str = f'{OUTPUT_FOLDER}/LOG'
if not Path(OUTPUT_FOLDER).is_dir():
    Path(OUTPUT_FOLDER).mkdir()
if not Path(LOG_FOLDER).is_dir():
    Path(LOG_FOLDER).mkdir()


def load_config_from_file(filename) -> Any | dict:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        console.print(f"Error loading configuration: {e}")
        return {}


def LF() -> None:
    return print('\n')


def set_env_variables() -> None:
    with open('config/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            os.environ[key.upper()] = str(value)


def excelConfigurator(config_keys) -> dict[str, str]:
    with open('config/columns.yaml', 'r', encoding='utf-8') as file:
        configurator = yaml.safe_load(file)
    return {key: configurator.get(key) for key in config_keys}


def regexConfigurator(regex_keys) -> dict[str, str]:
    with open('config/regex.yaml', 'r', encoding='utf-8') as file:
        regex = yaml.safe_load(file)
    return {key: regex.get(key) for key in regex_keys}


def load_kql_queries() -> Dict[str, Dict[str, str]]:
    with open('config/kql.json', 'r', encoding='utf-8') as file:  # Ensure correct file path and extension
        kql_queries = json.load(file)
    return kql_queries  # Return the loaded queries


class ExcelConfigurations(TypedDict):
    """The class `ExcelConfigurations` is a TypedDict that defines the structure of a configuration object for Excel files."""
    CL_INIT_ORDER: List[str]
    CL_DROPPED: List[str]
    CL_FINAL_ORDER: List[str]
    EXCLUSION_PAIRS: Dict[str, List[str]]


class RegexConfigurations(TypedDict):
    """The class `RegexConfigurations` is a TypedDict that defines the structure of a configuration object for Excel files."""
    AZDIAG_PATTS: List[str]
    AZDIAG_STRINGS: List[str]
    MATCH_VALUE_TO_SPECIFIC_COLUMN: Dict[str, List[str]]


class AppConfigurations(TypedDict):
    """The AppConfigurations class defines a dictionary type with specific keys and corresponding value types for storing application configurations."""
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
    AZDIAG_EXCEL_COMBINED_FILE: Path
    TEMP_EXCEL_FILE: Path
    Extraction_LogFILE: str
    TOKEN_URL: str
    AZLOG_ZEN_ENDPOINT: str


class LowerCaseMessageLogRecord(logging.LogRecord):

    def getMessage(self):
        original = super().getMessage()
        words = original.split()
        capitalized_words = [
            word.capitalize() if re.match(r'\bsaving\b', word, re.IGNORECASE) or re.match(r'\b(rows?|cols?)\b', word, re.IGNORECASE)
            or re.match(r'\b(regexs?|strings?|columns?|cols?|rows?|patterns?|totals?|removals?)\b', word, re.IGNORECASE) else word for word in words
        ]
        return ' '.join(capitalized_words)


class CustomRichHandler(RichHandler):

    def get_level_text(self, record) -> Text:
        log_level = record.levelname
        if log_level == "INFO":
            return Text(log_level, style="bold green")
        elif log_level == "WARNING":
            return Text(log_level, style="bold yellow")
        elif log_level == "ERROR":
            return Text(log_level, style="bold red")
        elif log_level == "DEBUG":
            return Text(log_level, style="bold blue")
        return super().get_level_text(record)


console = Console()  # Make console a global variable
install()
logging.setLogRecordFactory(LowerCaseMessageLogRecord)

keywords: list[str] = [
    "Regex",
    "bold blue",
    "String",
    "bold blue",
    "Row",
    "italic red",
    "Column",
    "italic red",
    "Col",
    "italic red",
    "Pattern",
    "italic green",
    "Total",
    "blue",
    "Removal",
    "bold red",
    "Missing",
    "bold red",
    "Removed",
    "red",
    "Dropped",
    "red",
    "Drop",
    "red",
    "Blank",
    "red",
    "Cannot",
    "red",
    "Key",
    "red",
    "Completed",
    "green",
    "Complete",
    "green",
    "Created",
    "green",
    "Formatted",
    "green",
    "Saved",
    "bold green",
    "Saving",
    "bold green",
    "Processing",
    "bold green",
    "File",
    "italic green",
    "Loaded",
    "bold green",
    "Not Found",
    "red",  #
    "not found",
    "red",
    "Found",
    "green",
    "Error",
    "bold red",
    "Warning",
    "bold red",
    "Info",
    "bold green",
    "Debug",
    "bold blue"
]


def setup_logging() -> None:
    global console  # pylint: disable=global-variable-not-assigned
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s [%(funcName)s]",
        datefmt="[%X]",
        handlers=[
            CustomRichHandler(
                level=logging.DEBUG,
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
                # locals_max_string=100,
                log_time_format="[%x %X]",
                keywords=keywords),
            logging.FileHandler('AZLOGS/LOG/applogs.log', mode='a', encoding='utf-8', delay=False, errors='ignore')
        ])


def jprint(data, level=logging.INFO):
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


# You can test this setup in this file itself if needed
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    file_handler = logging.getLogger().handlers[1]
    file_handler.setLevel(logging.DEBUG)
    logger.info("Logging setup is successful with RichHandler.")
    set_env_variables()
