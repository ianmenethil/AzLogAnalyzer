import logging
import sys
from typing import List, Literal, Dict, Tuple
from configurator import console, LF, load_config_from_file
from queryFormatter import QueryFormatter
from timeManager import TimeManager

logger = logging.getLogger(__name__)

get_current_time_info = TimeManager.get_current_time_info
is_date_string = TimeManager.is_date_string
convert_syd_to_aztime = TimeManager.convert_syd_to_aztime
validate_starttime_endtime = TimeManager.validate_starttime_endtime
convert_utc_to_sydney = TimeManager.convert_utc_to_sydney
validate_timeago_variable = TimeManager.validate_timeago_variable
input_datetime_with_validation = TimeManager.input_datetime_with_validation
time_info = get_current_time_info()
IP_FILEPATH: str = 'config/IPs.yaml'
TODAY: str = time_info['TODAY']
NOWINSYDNEY: str = time_info['NOWINSYDNEY']
NOWINAZURE: str = time_info['NOWINAZURE']
ip_data = load_config_from_file(IP_FILEPATH)
RAW_WHITELIST_IPs: List[str] = ip_data['WHITELIST']
WHITELIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_WHITELIST_IPs)
RAW_BLACKLIST_IPs: List[str] = ip_data['BLACKLIST']
BLACKLIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_BLACKLIST_IPs)
RAW_DYNAMIC_IPs: List[str] = ip_data['DYNAMICLIST']
DYNAMIC_IPs = ", ".join(f"'{ip}'" for ip in RAW_DYNAMIC_IPs)


class UserInputHandler():

    @staticmethod
    def list_choices() -> None:
        # Define query options
        query_options = {
            'AzureDiagnostics': {
                '1D': 'AZDIAG_DYNAMIC',
                '1A': 'AZDIAG_IP1IP2_TIMEAGO',
                '2A': 'AZDIAG_TIMEBETWEEN',
                '3A': 'AZDIAG_IP_TIMEBETWEEN',
                '4A': 'AZDIAG_TIMEAGO',
                '5A': 'AZDIAG_CUSTOM_TIMEAGO'
            },
            'AppRequests': {
                '1p': 'APPPAGE_TIMEAGO-notcoded'
            },
            'AppPages': {
                '1r': 'APPREQ_TIMEAGO'
            },
            'AppBrowser': {
                '1b': 'APPBROWSER_TIMEAGO'
            },
            'AppServiceHTTPLogs': {
                '1l': 'HTTPLogs_TIMEAGO',
                '2l': 'HTTPLogs_TIMEBETWEEN'
            },
            'AppServiceIPSec': {
                '1s': 'APPSERVIPSec_TIMEAGO'
            }
        }
        console.print('[red]Select query[/red]', style='bold', justify='center', markup=True)
        LF()
        for section, options in query_options.items():
            console.print('[yellow]#' * 20 + section + '#[/yellow]' * 20, style='bold', justify='center', markup=True)
            for key, value in options.items():
                console.print(f'{key}. [red]{value}[/red]', style='blink', justify='center', markup=True)
            LF()
        console.print('0. [magenta]Exit[/magenta]', style='blink', justify='center', markup=True)

    @staticmethod
    def select_query() -> str:
        """Selects a query from a list of options and returns the chosen query.
        Returns: str: The selected query."""
        UserInputHandler.list_choices()
        query = input('Select query: ').strip()
        query_options = {
            '1d': 'AZDIAG_DYNAMIC',
            '1a': 'AZDIAG_IP1IP2_TIMEAGO',
            '2a': 'AZDIAG_TIMEBETWEEN',
            '3a': 'AZDIAG_IP_TIMEBETWEEN',
            '4a': 'AZDIAG_TIMEAGO',
            '5a': 'AZDIAG_CUSTOM_TIMEAGO',
            '1p': 'APPPAGE_TIMEAGO',
            '1r': 'APPREQ_TIMEAGO',
            '1b': 'APPBROWSER_TIMEAGO',
            'l1': 'HTTPLogs_TIMEAGO',
            'l2': 'HTTPLogs_TIMEBETWEEN',
            '1s': 'APPSERVIPSec_TIMEAGO'
        }

        if query == '0':
            sys.exit()
        elif query in query_options:
            logger.info(f'Selected query: {query_options[query]}')
            return query_options[query]
        else:
            logger.info(f'Wrong Query selected: {query}')

            return UserInputHandler.select_query()

    @staticmethod
    def get_query_input() -> tuple:
        """Get user input for selecting and formatting a query.
        Returns: tuple: A tuple containing the selected query name and the formatted query content."""
        query_choice = UserInputHandler.select_query()
        query_name = query_choice
        ip1 = ip2 = None
        time_format_msg = f'Acceptable T format: {NOWINSYDNEY}'

        if query_name == "AZDIAG_IP1IP2_TIMEAGO":
            logger.info('AZDIAG_IP1IP2_TIMEAGO')
            ip1 = input('Enter value for IP1: ')
            ip2 = input('Enter value for IP2: ')
            timeago = input('AZDIAG_IP1IP2_TIMEAGO: ')
            while not validate_timeago_variable(timeago):
                logger.info('Include (m/h/d)')
                timeago = input('AZDIAG_IP1IP2_TIMEAGO: ')
            query_content = QueryFormatter.format_query(query_name, ip1=ip1, ip2=ip2, timeago=timeago, whitelist=WHITELIST_IPs)

        elif query_name == "AZDIAG_TIMEAGO":
            logger.info('AZDIAG_TIMEAGO')
            timeago = input('AZDIAG_TIMEAGO: ')
            while not validate_timeago_variable(timeago):
                logger.info('Include (m/h/d)')
                timeago = input('AZDIAG_TIMEAGO: ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago, whitelist=WHITELIST_IPs)

        elif query_name == "AZDIAG_CUSTOM_TIMEAGO":
            logger.info(f'AZDIAG_CUSTOM_TIMEAGO: {DYNAMIC_IPs}')
            timeago = input('AZDIAG_CUSTOM_TIMEAGO: Enter value for TIMEAGO : ')
            while not validate_timeago_variable(timeago):
                logger.info('Include (m/h/d)')
                timeago = input('Enter value for TIMEAGO : ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago, dynamiclist=DYNAMIC_IPs)

        elif query_name == "HTTPLogs_TIMEAGO":
            logger.info('HTTPLogs_TIMEAGO')
            timeago = input('HTTPLogs_TIMEAGO: Enter value for TIMEAGO : ')
            while not validate_timeago_variable(timeago):
                logger.info('Include (m/h/d)')
                timeago = input('Enter value for TIMEAGO : ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago, whitelist=WHITELIST_IPs)

        elif query_name == "AZDIAG_IP_TIMEBETWEEN":
            logger.info('AZDIAG_IP_TIMEBETWEEN')
            logger.info(f'Time in Sydney: {NOWINSYDNEY}')
            single_ip = input('Enter value for IP1: ')
            start_time = input_datetime_with_validation(f'AZDIAG_IP_TIMEBETWEEN {time_format_msg}: ')
            while not start_time:
                logger.info(f'{time_format_msg}: ')
                start_time = input_datetime_with_validation(f'AZDIAG_IP_TIMEBETWEEN {time_format_msg}: ')
            start_time = convert_syd_to_aztime(start_time)
            logger.info(f'Converted start_time: {start_time}')

            end_time = input_datetime_with_validation(f'AZDIAG_IP_TIMEBETWEEN {time_format_msg}: ')
            while not end_time:
                logger.info(f'{time_format_msg}: ')
                end_time = input_datetime_with_validation(f'AZDIAG_IP_TIMEBETWEEN {time_format_msg}: ')
            end_time = convert_syd_to_aztime(end_time)
            logger.info(f'Converted start_time: {end_time}')
            query_content = QueryFormatter.format_query(query_name, start_t=start_time, end_t=end_time, singleip=single_ip, whitelist=WHITELIST_IPs)

        elif query_name == "AZDIAG_TIMEBETWEEN":
            logger.info(f'Time in Sydney: {NOWINSYDNEY}')
            logger.info(f'AZDIAG_TIMEBETWEEN {time_format_msg}')
            start_time = input_datetime_with_validation(f'AZDIAG_TIMEBETWEEN {time_format_msg}: ')
            while not start_time:
                logger.info(f'{time_format_msg}: ')
                start_time = input_datetime_with_validation(f'AZDIAG_TIMEBETWEEN {time_format_msg}: ')
            start_time = convert_syd_to_aztime(start_time)
            logger.info(f'Converted start_time: {start_time}')

            end_time = input_datetime_with_validation(f'AZDIAG_TIMEBETWEEN {time_format_msg}: ')
            while not end_time:
                logger.info(f'{time_format_msg}: ')
                end_time = input_datetime_with_validation(f'AZDIAG_TIMEBETWEEN {time_format_msg}: ')
            end_time = convert_syd_to_aztime(end_time)
            logger.info(f'Converted start_time: {end_time}')

            query_content = QueryFormatter.format_query(query_name, start_t=start_time, end_t=end_time, whitelist=WHITELIST_IPs)

        elif query_name == "HTTPLogs_TIMEBETWEEN":
            logger.info('HTTPLogs_TIMEBETWEEN')
            logger.info(f'Time in Sydney: {NOWINSYDNEY}')
            start_time = input_datetime_with_validation('HTTPLogs_TIMEBETWEEN {time_format_msg}: ')
            while not start_time:
                logger.info(f'{time_format_msg}: ')
                start_time = input_datetime_with_validation('HTTPLogs_TIMEBETWEEN {time_format_msg}: ')
            start_time = convert_syd_to_aztime(start_time)
            logger.info(f'Converted start_time: {start_time}')

            end_time = input_datetime_with_validation('HTTPLogs_TIMEBETWEEN {time_format_msg}: ')
            while not end_time:
                logger.info(f'{time_format_msg}: ')
                end_time = input_datetime_with_validation('HTTPLogs_TIMEBETWEEN {time_format_msg}: ')
            end_time = convert_syd_to_aztime(end_time)
            logger.info(f'Converted start_time: {end_time}')
            query_content = QueryFormatter.format_query(query_name, start_t=start_time, end_t=end_time, whitelist=WHITELIST_IPs)

        elif query_name == "APPREQ_TIMEAGO":
            logger.info('APPREQ_TIMEAGO')
            logger.info(f'Current Date: {TODAY}')
            timeago = input('APPREQ_TIMEAGO: ')
            while not validate_timeago_variable(timeago):
                logger.info('Include (m/h/d)')
                timeago = input('Enter value for TIMEAGO : ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago)
            logger.info(f'query_content: {query_content}')

        elif query_name == "AZDIAG_DYNAMIC":
            logger.info('AZDIAG_DYNAMIC')
            dynamic_params = UserInputHandler.dynamic_query_input()
            logger.info(f'Input: {dynamic_params}')
            logger.info(f'choice {query_choice}')
            logger.info(f'query name: {query_name}')
            logger.info(f'dynamic_params: {dynamic_params}')

            query_content = QueryFormatter.format_dynamic_query(query_name, **dynamic_params)
            logger.info(f'query_content: {query_content}, {type(query_content)}, {query_content[1]}, {type(query_content[1])} dynamic_params: {dynamic_params}')
            return query_name, query_content

        else:
            logger.error('Else: section ', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
            sys.exit()
        return query_name, query_content

    @staticmethod
    def dynamic_query_input():
        print("Select time condition:")
        print("1. Time Ago")
        print("2. Time Between")
        time_choice = input("Choice (number): ")

        query_params = {}
        time_format_msg = f'Acceptable T format {NOWINSYDNEY}'

        if time_choice == '1':
            logger.info(f'Current Date: {TODAY}')
            q_tago = input("Enter value for TIMEAGO: ")
            query_params['timeago'] = q_tago
            while not validate_timeago_variable(q_tago):
                logger.info('Include (m/h/d)')
                timeago = input('Enter value for TIMEAGO: ')
                query_params['timeago'] = timeago
                logger.info(f'query_params: {query_params}')
            query_params['timeago'] = q_tago
            logger.info(f'query_params: {query_params}')

        elif time_choice == '2':
            logger.info(NOWINSYDNEY)
            start_t = input(f"{time_format_msg} Enter start time: ")
            query_params['start_t'] = start_t
            while not start_t:
                logger.info(f'{time_format_msg}: ')
                start_t = input_datetime_with_validation(f'AZDIAG_DYNAMIC{time_format_msg}: ')
            start_t = convert_syd_to_aztime(start_t)
            logger.info(f'Converted start_t: {start_t}')
            query_params['start_t'] = start_t

            end_t = input_datetime_with_validation(f'AZDIAG_DYNAMIC {time_format_msg}: ')

            end_t = input("Enter end time (DD-MM-YY HH:MM): ")
            while not end_t:
                logger.info(f'{time_format_msg}: ')
                end_t = input_datetime_with_validation(f'AZDIAG_DYNAMIC {time_format_msg}: ')
            end_t = convert_syd_to_aztime(end_t)
            logger.info(f'Converted start_t: {end_t}')
            query_params['end_t'] = end_t
            # end_t = query_params['end_t']
            logger.info(f'query_params: {query_params}')
            input()

        return query_params

        # ! AZDIAG_TIMEAGO

    @staticmethod
    def get_time_input():
        print("Select your time condition:")
        print("1. Time Ago")
        print("2. Time Between")
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            timeago = input("Enter the time ago  ")
            return {'time_condition': 'timeago', 'timeago': timeago}
        elif choice == '2':
            start_t = input("Enter the start time (DD-MM-YY HH:MM): ")
            end_t = input("Enter the end time (DD-MM-YY HH:MM): ")
            return {'time_condition': 'timebetween', 'start_t': start_t, 'end_t': end_t}
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return UserInputHandler.get_time_input()

    # @staticmethod
    # def handle_query_selection():
    #     # Example function to demonstrate the flow
    #     time_params = UserInputHandler.get_time_input()
    #     query_name = "KQL_AZDIAG_DYNAMIC"  # Or however you determine the appropriate query to use
    #     formatted_query = QueryFormatter.format_query(query_name, **time_params)
    #     print(formatted_query)
