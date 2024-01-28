# configurator.py
import os
import json
import logging
from typing import Any, List, Dict, Tuple, Union, TypedDict, Optional  # pylint: disable=unused-import
import functools  # pylint: disable=unused-import
import inspect  # pylint: disable=unused-import
import traceback  # pylint: disable=unused-import
import yaml
from rich.console import Console
from rich.traceback import install, Traceback  # pylint: disable=unused-import
from rich.pretty import pprint  # pylint: disable=unused-import
from rich.logging import RichHandler
from rich.pretty import Pretty
from rich.text import Text
import re

COLUMN_CONFIG_FILE = 'config/column_config.yaml'
CONFIG_FILE = 'config/config.yaml'

# def DEC_MOREDETAILS(func):

#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             caller = inspect.stack()[1]
#             console.print(f"[bold cyan]Function {func.__name__} was called from:[/bold cyan]")
#             console.print(f"Filename: {caller.filename}")
#             console.print(f"Function: {caller.function}")
#             console.print(f"Line: {caller.lineno}")
#             console.print("Event: function_call")
#             result = func(*args, **kwargs)
#             console.print(f"[bold green]Function {func.__name__} executed successfully.[/bold green]")
#             return result
#         except Exception as e:
#             traceback = Traceback.from_exception(e, exc_value=e, traceback=traceback)
#             console.print(traceback)

#     return wrapper


# @DEC_MOREDETAILS
def load_config_from_file() -> Any | dict:
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        console.print(f"Error loading configuration: {e}")
        return {}


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


class LowerCaseMessageLogRecord(logging.LogRecord):

    def getMessage(self):
        original = super().getMessage()
        words = original.split()
        capitalized_words = [
            word.capitalize() if re.match(r'\bsaving\b', word, re.IGNORECASE) or re.match(r'\b(rows?|cols?)\b', word, re.IGNORECASE) or
            re.match(r'\b(regexs?|strings?|columns?|cols?|rows?|patterns?|totals?|removals?)\b', word, re.IGNORECASE) else word for word in words
        ]
        return ' '.join(capitalized_words)


class CustomRichHandler(RichHandler):

    def get_level_text(self, record) -> Text:
        log_level = record.levelname
        if log_level == "INFO":
            return Text(log_level, style="bold green")
        elif log_level == "WARNING":
            return Text(log_level, style="bold red")
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
                # locals_max_string=100,
                log_time_format="[%x %X]",
                keywords=keywords),
            logging.FileHandler('AZLOGS/LOG/applogs.log', mode='w', encoding='utf-8', delay=False, errors='ignore')
        ])


def jprint(data, level=logging.INFO):
    global console
    try:
        pretty = Pretty(data)
        console.print(pretty)
        if level == logging.DEBUG:
            logger = logging.getLogger(__name__)
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
