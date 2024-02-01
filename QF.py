import logging
from typing import Dict
from configurator import load_kql_queries
# from configLoader import ConfigLoader

logger = logging.getLogger(__name__)

WHITELIST_IPs = ['1.1.1.1', '2.2.2.2']

kql_queries: Dict[str, Dict[str, str]] = load_kql_queries()

raw_kql_queries = {
    "KQL_AZDIAG_IP1IP2_TIMEAGO": kql_queries["F_AzDiag_IP1IP2TIMEAGO"]["query"],
    "KQL_AZDIAG_TIMEBETWEEN": kql_queries["F_AzDiag_TIMEBETWEEN"]["query"],
    "KQL_AZDIAG_IP_TIMEBETWEEN": kql_queries["F_AzDiag_IP_TIMEBETWEEN"]["query"],
    "KQL_AZDIAG_TIMEAGO": kql_queries["F_AzDiag_TIMEAGO"]["query"],
    "KQL_APPREQ_TIMEAGO": kql_queries["F_AppReq_TIMEAGO"]["query"],
    "KQL_APPPAGE_TIMEAGO": kql_queries["F_AppPage_TIMEAGO"]["query"],
    "KQL_APPBROWSER_TIMEAGO": kql_queries["F_AppBrowser_TIMEAGO"]["query"],
    "KQL_HTTPLogs_TIMEAGO": kql_queries["F_HTTPLogs_TIMEAGO"]["query"],
    "KQL_HTTPLogs_TIMEBETWEEN": kql_queries["F_HTTPLogs_TIMEBETWEEN"]["query"],
    "KQL_APPSERVIPSec_TIMEAGO": kql_queries["F_AppServIPSec_TIMEAGO"]["query"],
}


class QueryFormatter():

    @staticmethod
    def format_query(query_name, ip1=None, ip2=None, timeago=None, iplist=None, start_t=None, end_t=None, whitelist=None, blacklist=None, singleip=None) -> str:  # pylint: disable=unused-argument # noqa: W0613
        """Formats a query based on the given parameters.
        Parameters:
        - query_name: The name of the query.
        - ip1: The value for IP1 (optional).
        - ip2: The value for IP2 (optional).
        - timeago: The value for TIMEAGO (optional).
        - iplist: The list of IPs (optional).
        - start_t: The start time (optional).
        - end_t: The end time (optional).
        Returns:
        - str: The formatted query."""
        query_param_map = {
            "AZDIAG_IP1IP2_TIMEAGO": {
                "IP1_INFILE": ip1,
                "IP2_INFILE": ip2,
                "TIME_INFILE": timeago,
                "WHITE_INFILE": WHITELIST_IPs,
            },
            "AZDIAG_TIMEBETWEEN": {
                "STARTTIME_INFILE": start_t,
                "ENDTIME_INFILE": end_t,
                "WHITE_INFILE": WHITELIST_IPs,
            },
            "AZDIAG_IP_TIMEBETWEEN": {
                "STARTTIME_INFILE": start_t,
                "ENDTIME_INFILE": end_t,
                "SINGLEIP_INFILE": singleip,
                "WHITE_INFILE": WHITELIST_IPs,
            },
            "AZDIAG_TIMEAGO": {
                "TIME_INFILE": timeago,
                "WHITE_INFILE": WHITELIST_IPs
            },
            "APPREQ_TIMEAGO": {
                "TIME_INFILE": timeago
            },
            "APPPAGE_TIMEAGO": {
                "TIME_INFILE": timeago
            },
            "APPBROWSER_TIMEAGO": {
                "TIME_INFILE": timeago
            },
            "HTTPLogs_TIMEAGO": {
                "TIME_INFILE": timeago
            },
            "HTTPLogs_TIMEBETWEEN": {
                "STARTTIME_INFILE": start_t,
                "ENDTIME_INFILE": end_t
            },
            "APPSERVIPSecTIMEAGO": {
                "TIME_INFILE": timeago
            },
        }

        formatted_query_name = f"KQL_{query_name}"
        if formatted_query_name in raw_kql_queries:
            query_template = raw_kql_queries[formatted_query_name]
            query_params = query_param_map.get(query_name, {})
            return query_template.format(**query_params)
        logger.error(f'Active [yellow]Query Name:[/yellow] [red]{query_name}[/red] not found\nExiting...', exc_info=True, stack_info=True, extra={'markup': True})
        return 'Not Found'
