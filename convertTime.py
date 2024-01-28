# convertTime.py
# from datetime import datetime
import pytz
from dateutil import parser


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


# def convert_utc_to_sydney(utc_timestamp_str):
#     try:
#         # Create a timezone object for UTC and Sydney
#         utc_zone = pytz.utc
#         sydney_zone = pytz.timezone('Australia/Sydney')

#         # Check if the input string contains microseconds
#         if '.' in utc_timestamp_str:
#             # Trim the timestamp to the right length for microseconds (6 digits)
#             parts = utc_timestamp_str.split('.')
#             microsecond_part = parts[1][:6]  # Take only the first 6 digits
#             utc_timestamp_str = parts[0] + '.' + microsecond_part + 'Z'
#             # Parse the UTC timestamp with microseconds and set its timezone to UTC
#             utc_dt = datetime.strptime(utc_timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=utc_zone)
#         else:
#             # Parse the UTC timestamp without microseconds and set its timezone to UTC
#             utc_dt = datetime.strptime(utc_timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc_zone)

#         # Convert the timestamp to Sydney local time
#         sydney_dt = utc_dt.astimezone(sydney_zone)

#         # return sydney_dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
#         # return sydney_dt.strftime("%d-%m-%y %H:%M")
#         return sydney_dt.strftime("%d-%m-%y %H:%M:%S")
#     except ValueError as e:
#         print(f"Invalid input format: {e}")
#         return None

#     except Exception as e:
#         print(f"Error: {e}")
#         return None
