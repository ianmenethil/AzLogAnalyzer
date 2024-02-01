import logging
import sys
from configurator import console, LF
from QF import QueryFormatter
from timeManager import TimeManager

logger = logging.getLogger(__name__)

get_current_time_info = TimeManager.get_current_time_info
is_date_string = TimeManager.is_date_string
convert_syd_to_aztime = TimeManager.convert_syd_to_aztime
validate_starttime_endtime = TimeManager.validate_starttime_endtime
convert_utc_to_sydney = TimeManager.convert_utc_to_sydney
validate_timeago_variable = TimeManager.validate_timeago_variable
input_datetime_with_validation = TimeManager.input_datetime_with_validation
WHITELIST_IPs = ['1.1.1.1', '2.2.2.2']

time_info = get_current_time_info()
TODAY: str = time_info['TODAY']


class UserInputHandler():

    @staticmethod
    def list_choices() -> None:
        console.print('[red]Select query[/red]', style='bold', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AzureDiagnostics' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print('1. [blue]AZDIAG_IP1IP2_TIMEAGO[/blue]', style='italic', justify='center', markup=True)
        console.print('2. [blue]AZDIAG_TIMEBETWEEN[/blue]', style='blink', justify='center', markup=True)
        console.print('3. [blue]AZDIAG_IP_TIMEBETWEEN[/blue]', style='blink', justify='center', markup=True)
        console.print('4. [blue]AZDIAG_TIMEAGO[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppRequests' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print('5. [blue]APPREQ_TIMEAGO - not yet[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppPages' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print('6. [blue]APPPAGE_TIMEAGO - not yet[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppBrowser' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print('7. [blue]APPBROWSER_TIMEAGO - not yet[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppServiceHTTPLogs' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print('8. [blue]HTTPLogs_TIMEAGO[/blue]', style='blink', justify='center', markup=True)
        console.print('9. [blue]HTTPLogs_TIMEBETWEEN[/blue]', style='blink', justify='center', markup=True)
        LF()
        console.print('[yellow]#' * 30 + 'AppServiceIPSec' + '#[/yellow]' * 30, style='bold', justify='center', markup=True)
        console.print('10. [blue]APPSERVIPSec_TIMEAGO - not yet[/blue]', style='blink', justify='center', markup=True)
        console.print('0. [magenta]Exit[/magenta]', style='blink', justify='center', markup=True)

    @staticmethod
    def select_query():
        """Selects a query from a list of options and returns the chosen query."""
        UserInputHandler.list_choices()
        query = input('Select query: ')
        query = query.strip()
        if query == '1':
            return 'AZDIAG_IP1IP2_TIMEAGO'
        elif query == '2':
            return 'AZDIAG_TIMEBETWEEN'
        elif query == '3':
            return 'AZDIAG_IP_TIMEBETWEEN'
        elif query == '4':
            return 'AZDIAG_TIMEAGO'
        elif query == '5':
            return 'APPREQ_TIMEAGO'
        elif query == '6':
            return 'APPPAGE_TIMEAGO'
        elif query == '7':
            return 'APPBROWSER_TIMEAGO'
        elif query == '8':
            return 'HTTPLogs_TIMEAGO'
        elif query == '9':
            return 'HTTPLogs_TIMEBETWEEN'
        elif query == '10':
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
            logger.info(f'Current date: {TODAY}')
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
