"""Simple MCP server providing datetime tools — for testing the MCP pipeline."""

import datetime
import zoneinfo

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("datetime")


@mcp.tool()
def get_current_time(timezone: str = "Asia/Tokyo") -> str:
    """Get the current date and time in the specified timezone.

    Args:
        timezone: IANA timezone name (e.g. "Asia/Tokyo", "UTC", "US/Eastern")
    """
    tz = zoneinfo.ZoneInfo(timezone)
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@mcp.tool()
def get_day_of_week(date: str) -> str:
    """Get the day of the week for a given date.

    Args:
        date: Date in YYYY-MM-DD format
    """
    d = datetime.date.fromisoformat(date)
    days_ja = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
    days_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    idx = d.weekday()
    return f"{days_en[idx]} ({days_ja[idx]})"


if __name__ == "__main__":
    mcp.run(transport="stdio")
