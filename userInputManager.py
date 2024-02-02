import logging

logger = logging.getLogger(__name__)
import sys
from typing import List
from configurator import console, LF, load_config_from_file
from QF import QueryFormatter
from timeManager import TimeManager

get_current_time_info = TimeManager.get_current_time_info
is_date_string = TimeManager.is_date_string
convert_syd_to_aztime = TimeManager.convert_syd_to_aztime
validate_starttime_endtime = TimeManager.validate_starttime_endtime
convert_utc_to_sydney = TimeManager.convert_utc_to_sydney
validate_timeago_variable = TimeManager.validate_timeago_variable
input_datetime_with_validation = TimeManager.input_datetime_with_validation
time_info = get_current_time_info()
TODAY: str = time_info['TODAY']
NOWINSYDNEY: str = time_info['NOWINSYDNEY']
NOWINAZURE: str = time_info['NOWINAZURE']
# # ? IPs
IP_FILEPATH: str = 'config/IPs.yaml'
ip_data = load_config_from_file(IP_FILEPATH)
RAW_WHITELIST_IPs: List[str] = ip_data['WHITELIST']
WHITELIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_WHITELIST_IPs)

RAW_BLACKLIST_IPs: List[str] = ip_data['BLACKLIST']
BLACKLIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_BLACKLIST_IPs)
RAW_DYNAMIC_IPs: List[str] = ip_data['DYNAMICLIST']
DYNAMIC_IPs = ", ".join(f"'{ip}'" for ip in RAW_DYNAMIC_IPs)

# logger.info(f"WHITELIST: {WHITELIST_IPs}")
# logger.info(f"BLACKLIST: {BLACKLIST_IPs}")
# logger.info(f"DYNAMIC_IPs: {DYNAMIC_IPs}")


class UserInputHandler():

    @staticmethod
    def list_choices() -> None:
        a1 = 'AZDIAG_IP1IP2_TIMEAGO'
        a2 = 'AZDIAG_TIMEBETWEEN'
        a3 = 'AZDIAG_IP_TIMEBETWEEN'
        a4 = 'AZDIAG_TIMEAGO'
        a5 = 'AZDIAG_CUSTOM_TIMEAGO'
        # a11 = 'APPREQ_TIMEAGO'
        p1 = 'APPPAGE_TIMEAGO'
        r1 = 'APPREQ_TIMEAGO'
        b1 = 'APPBROWSER_TIMEAGO'
        l1 = 'HTTPLogs_TIMEAGO'
        l2 = 'HTTPLogs_TIMEBETWEEN'
        s1 = 'APPSERVIPSec_TIMEAGO'
        console.print('[red]Select query[/red]', style='bold', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AzureDiagnostics' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print(f'1a. [red]{a1}[/red]', style='blink', justify='center', markup=True)
        console.print(f'2a. [red]{a2}[/red]', style='blink', justify='center', markup=True)
        console.print(f'3a. [red]{a3}[/red]', style='blink', justify='center', markup=True)
        console.print(f'4a. [red]{a4}[/red]', style='blink', justify='center', markup=True)
        console.print(f'5a. [red]{a5}[/red]', style='blink', justify='center', markup=True)
        # console.print(f'11a. [red]{a11}[/red]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppRequests' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print(f'1p. [blue]{p1} - not yet[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppPages' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print(f'1r. [blue]{r1} - not yet[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppBrowser' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print(f'1b. [blue]{b1} - not yet[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppServiceHTTPLogs' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print(f'1l. [blue]{l1}/blue]', style='blink', justify='center', markup=True)
        console.print(f'2l. [blue]{l2}[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppServiceIPSec' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print(f'1s. [blue]{s1} - not yet[/blue]', style='blink', justify='center', markup=True)
        console.print('0. [magenta]Exit[/magenta]', style='blink', justify='center', markup=True)

    @staticmethod
    def select_query():
        """Selects a query from a list of options and returns the chosen query."""
        UserInputHandler.list_choices()
        query = input('Select query: ')
        query = query.strip()
        if query == '1a':
            return 'AZDIAG_IP1IP2_TIMEAGO'
        elif query == '2a':
            return 'AZDIAG_TIMEBETWEEN'
        elif query == '3a':
            return 'AZDIAG_IP_TIMEBETWEEN'
        elif query == '4a':
            return 'AZDIAG_TIMEAGO'
        elif query == '5a':
            return 'AZDIAG_CUSTOM_TIMEAGO'
        elif query == '1p':
            return 'APPPAGE_TIMEAGO'
        elif query == '1r':
            return 'APPREQ_TIMEAGO'
        elif query == '1b':
            return 'APPBROWSER_TIMEAGO'
        elif query == 'l1':
            return 'HTTPLogs_TIMEAGO'
        elif query == 'l2':
            return 'HTTPLogs_TIMEBETWEEN'
        elif query == '1s':
            return 'APPSERVIPSec_TIMEAGO'
        elif query == '0':
            sys.exit()
        logger.info(f'Query selected: {query}')
        logger.info('Wrong query, try again')
        return UserInputHandler.select_query()

    @staticmethod
    def get_query_input():  # ? 1
        """Get user input for selecting and formatting a query.
        Returns:tuple: A tuple containing the selected query name and the formatted query content."""
        query_choice = UserInputHandler.select_query()
        query_name = query_choice
        ip1 = ip2 = None
        time_format_msg = '\nAcceptable formats\nDD-MM-YY HH:MM or DD-MM-YY HH:MM:SS\n'
        time_format_msg += f'{NOWINSYDNEY}'
        # ! AZDIAG_IP1IP2_TIMEAGO
        if query_name == "AZDIAG_IP1IP2_TIMEAGO":
            logger.info('AZDIAG_IP1IP2_TIMEAGO')
            ip1 = input('Enter value for IP1: ')
            ip2 = input('Enter value for IP2: ')
            timeago = input('AZDIAG_IP1IP2_TIMEAGO: Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            while not validate_timeago_variable(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, ip1=ip1, ip2=ip2, timeago=timeago, whitelist=WHITELIST_IPs)

        # ! AZDIAG_TIMEAGO
        elif query_name == "AZDIAG_TIMEAGO":
            logger.info('AZDIAG_TIMEAGO')
            timeago = input('AZDIAG_TIMEAGO: Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            while not validate_timeago_variable(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago, whitelist=WHITELIST_IPs)

        # ! F_AzDiag_CUSTOM_TIMEAGO
        elif query_name == "AZDIAG_CUSTOM_TIMEAGO":
            logger.info(f'AZDIAG_CUSTOM_TIMEAGO: {DYNAMIC_IPs}')
            timeago = input('AZDIAG_CUSTOM_TIMEAGO: Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            while not validate_timeago_variable(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago, dynamiclist=DYNAMIC_IPs)
        # ! HTTPLogs_TIMEAGO
        elif query_name == "HTTPLogs_TIMEAGO":
            logger.info('HTTPLogs_TIMEAGO')
            timeago = input('HTTPLogs_TIMEAGO: Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            while not validate_timeago_variable(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago, whitelist=WHITELIST_IPs)

        # ! AZDIAG_IP_TIMEBETWEEN
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

        # ! AZDIAG_TIMEBETWEEN
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

        # ! HTTPLogs_TIMEBETWEEN
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

        else:
            console.print('Else: section ', style='bold', justify='center', markup=True)
            sys.exit()
            timeago = input('Else: Enter value for TIMEAGO(10m) (2h) (1d): ')
            while not validate_timeago_variable(timeago):
                logger.info('Invalid format. Please include (m)inutes, (h)ours, (d)ays, or (w)eeks. E.g., "10m" for 10 minutes.')
                timeago = input('Enter value for TIMEAGO (e.g., 10m, 2h, 1d): ')
            query_content = QueryFormatter.format_query(query_name, timeago=timeago)
            sys.exit()
        return query_name, query_content
