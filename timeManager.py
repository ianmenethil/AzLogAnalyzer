# time_ut.py
# from configurator import setup_logging
# setup_logging()
from datetime import datetime
from typing import Any
import re
import logging
import pytz
from dateutil import parser

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.info('Imported time_ut')

TODAY = datetime.now().strftime('%d-%m-%y')


class TimeManager:

    @staticmethod
    def get_current_time_info() -> dict[str, Any]:
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
    def convert_utc_to_sydney(utc_timestamp_str):
        try:
            # Define the Sydney timezone
            sydney_zone = pytz.timezone('Australia/Sydney')
            # Parse the UTC timestamp string to a datetime object automatically
            utc_dt = parser.parse(utc_timestamp_str)
            # Convert the timestamp to Sydney local time
            sydney_dt = utc_dt.astimezone(sydney_zone)
            # Format and return the Sydney time as a string
            return sydney_dt.strftime("%d-%m-%y %H:%M:%S")
        except ValueError as e:
            print(f"Invalid input format: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    @staticmethod
    def convert_syd_to_aztime(time_str):
        """Converts a Sydney time string to an Azure time string. Should be converted to "2024-02-01T05:44:34.0000000Z" format"""
        sydney_zone = pytz.timezone('Australia/Sydney')
        utc_zone = pytz.utc
        try:
            # Try to parse the time string with seconds
            sydney_now = datetime.strptime(time_str, '%d-%m-%y %H:%M:%S')
        except ValueError:
            # If that fails, try to parse it without seconds
            sydney_now = datetime.strptime(time_str, '%d-%m-%y %H:%M')
        sydney_now = sydney_zone.localize(sydney_now)
        utc_now = sydney_now.astimezone(utc_zone)
        # Set microseconds to 0
        utc_now = utc_now.replace(microsecond=0)
        # Format the time string in the desired format
        return utc_now.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')

    @staticmethod
    def validate_starttime_endtime(time_str) -> str:
        """Validate if the input string matches Azure's expected date-time formats."""
        try:
            # Attempt to parse the datetime string
            logger.info(f'Received time string: {time_str}\n Parsing...\n')
            az = TimeManager.convert_syd_to_aztime(time_str)
            logger.info(f'Converted time: {az}')
            return az
        except ValueError as e:
            return f'Error in az conversion {e}'

    @staticmethod
    def input_datetime_with_validation(prompt) -> str:
        """Prompt the user for a datetime input and validate it to a proper format."""
        while True:
            input_time = input(prompt)
            if re.match(r'^\d{2}-\d{2}-(\d{2}|\d{4}) \d{2}:\d{2}(-\d{2})?$', input_time):
                try:
                    datetime.strptime(input_time, '%d-%m-%y %H:%M')
                    return input_time
                except ValueError:
                    logger.info("Invalid date or time.")
            else:
                logger.info("Format must be: DD-MM-YY HH:MM or DD-MM-YY HH:MM-SS")

    @staticmethod
    def validate_timeago_variable(time_str) -> bool:
        """Validates if the time string includes a time unit (m, h, d, w). Returns True if valid, False otherwise."""
        return bool(re.match(r'^\d+[mhdw]$', time_str))

    @staticmethod
    def is_date_string(string, date_format) -> bool:  # for backups
        """Check if the string s matches the date_format."""
        try:
            datetime.strptime(string, date_format)
            return True
        except ValueError:
            return False
