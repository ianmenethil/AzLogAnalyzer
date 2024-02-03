# queryFormatter.py
import logging
from typing import Dict, List
from configurator import load_kql_queries, load_config_from_file
# from configLoader import ConfigLoader

logger = logging.getLogger(__name__)
# # ? IPs
IP_FILEPATH: str = 'config/IPs.yaml'
ip_data = load_config_from_file(IP_FILEPATH)
RAW_WHITELIST_IPs: List[str] = ip_data['WHITELIST']
WHITELIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_WHITELIST_IPs)
RAW_BLACKLIST_IPs: List[str] = ip_data['BLACKLIST']
BLACKLIST_IPs = ", ".join(f"'{ip}'" for ip in RAW_BLACKLIST_IPs)
RAW_DYNAMIC_IPs: List[str] = ip_data['DYNAMICLIST']
DYNAMIC_IPs = ", ".join(f"'{ip}'" for ip in RAW_DYNAMIC_IPs)

logger.info(f"WHITELIST: {WHITELIST_IPs}")
logger.info(f"BLACKLIST: {BLACKLIST_IPs}")
logger.info(f"DYNAMIC_IPs: {DYNAMIC_IPs}")

kql_queries: Dict[str, Dict[str, str]] = load_kql_queries()

raw_kql_queries = {
    "KQL_AZDIAG_IP1IP2_TIMEAGO": kql_queries["F_AzDiag_IP1IP2TIMEAGO"]["query"],
    "KQL_AZDIAG_TIMEBETWEEN": kql_queries["F_AzDiag_TIMEBETWEEN"]["query"],
    "KQL_AZDIAG_IP_TIMEBETWEEN": kql_queries["F_AzDiag_IP_TIMEBETWEEN"]["query"],
    "KQL_AZDIAG_TIMEAGO": kql_queries["F_AzDiag_TIMEAGO"]["query"],
    "KQL_AZDIAG_CUSTOM_TIMEAGO": kql_queries["F_AzDiag_CUSTOM_TIMEAGO"]["query"],
    "KQL_APPREQ_TIMEAGO": kql_queries["F_AppReq_TIMEAGO"]["query"],
    "KQL_APPPAGE_TIMEAGO": kql_queries["F_AppPage_TIMEAGO"]["query"],
    "KQL_APPBROWSER_TIMEAGO": kql_queries["F_AppBrowser_TIMEAGO"]["query"],
    "KQL_HTTPLogs_TIMEAGO": kql_queries["F_HTTPLogs_TIMEAGO"]["query"],
    "KQL_HTTPLogs_TIMEBETWEEN": kql_queries["F_HTTPLogs_TIMEBETWEEN"]["query"],
    "KQL_APPSERVIPSec_TIMEAGO": kql_queries["F_AppServIPSec_TIMEAGO"]["query"],
    "KQL_AZDIAG_DYNAMIC": kql_queries["AZDIAG_DYNAMIC"]["query"],
}
# # print KQL_AZDIAG_DYNAMIC
# logger.info(raw_kql_queries["KQL_AZDIAG_DYNAMIC"])
# input()


class QueryFormatter():

    @staticmethod
    def format_dynamic_query(query_name, **kwargs):
        # logger.info(f"Available queries: {list(kql_queries.keys())}")
        query_template = kql_queries.get(query_name, {}).get("query", "Not Found")
        if query_template == "Not Found":
            logger.error(f"Query Name: {query_name} not found")
            return query_template
        conditions = []
        if 'timeago' in kwargs:
            conditions.append(f"TimeGenerated >= ago({kwargs['timeago']})")
        elif 'start_t' in kwargs and 'end_t' in kwargs:
            conditions.append(f"TimeGenerated between(datetime('{kwargs['start_t']}') .. datetime('{kwargs['end_t']}'))")

        formatted_query = query_template.format(TIMEOPTION=" | ".join(conditions), WHITE_INFILE=WHITELIST_IPs)

        logger.debug(f"Formatted query: {formatted_query}", stacklevel=2)
        return formatted_query

    try:

        @staticmethod
        def format_query(query_name,
                         ip1=None,
                         ip2=None,
                         timeago=None,
                         iplist=None,
                         start_t=None,
                         end_t=None,
                         whitelist=None,
                         blacklist=None,
                         singleip=None,
                         dynamiclist=None,
                         timeoption=None) -> str:  # pylint: disable=unused-argument # noqa: W0613
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
                "AZDIAG_CUSTOM_TIMEAGO": {
                    "TIME_INFILE": timeago,
                    "DYNAMICLIST_INFILE": DYNAMIC_IPs
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
                "AZDIAG_DYNAMIC": {
                    "TIMEOPTION_INFILE": timeoption,
                    "DYNAMICLIST_INFILE": DYNAMIC_IPs
                },
            }
            logger.debug(f"Query param map: {query_param_map}", stacklevel=2)
            logger.debug(f"Query name: {query_name}", stacklevel=2)
            formatted_query_name = f"KQL_{query_name}"
            logger.warning(f"Formatted query name: {formatted_query_name}")
            if formatted_query_name in raw_kql_queries:
                query_template = raw_kql_queries[formatted_query_name]
                query_params = query_param_map.get(query_name, {})
                return query_template.format(**query_params)
            logger.error(f'Active [yellow]Query Name:[/yellow] [red]{query_name}[/red] not found\nExiting...', exc_info=True, stack_info=True, extra={'markup': True})
            return 'Not Found'
    except Exception as e:
        logger.error(e, exc_info=True, stack_info=True, extra={'markup': True})
