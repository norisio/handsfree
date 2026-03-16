import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from adb_shell.adb_device_async import AdbDeviceTcpAsync
from adb_shell.auth.keygen import keygen
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
from mcp.server.fastmcp import FastMCP
from wakeonlan import send_magic_packet

TV_IP = "192.168.10.211"
TV_MAC = "14:50:51:52:44:77"
ADB_PORT = 5555
KEY_PATH = Path(__file__).parent / "adbkey"

# チャンネルボタン 1-12 に対応するキーコード
CHANNEL_KEYCODES = {
    1: "KEYCODE_1",
    2: "KEYCODE_2",
    3: "KEYCODE_3",
    4: "KEYCODE_4",
    5: "KEYCODE_5",
    6: "KEYCODE_6",
    7: "KEYCODE_7",
    8: "KEYCODE_8",
    9: "KEYCODE_9",
    10: "KEYCODE_0",
    11: "KEYCODE_STAR",
    12: "KEYCODE_POUND",
}


def _get_signer() -> PythonRSASigner:
    if not KEY_PATH.exists():
        keygen(str(KEY_PATH))
    with open(KEY_PATH) as f:
        priv = f.read()
    with open(str(KEY_PATH) + ".pub") as f:
        pub = f.read()
    return PythonRSASigner(pub, priv)


async def _connect() -> AdbDeviceTcpAsync:
    device = AdbDeviceTcpAsync(TV_IP, ADB_PORT)
    await device.connect(rsa_keys=[_get_signer()], auth_timeout_s=10)
    return device


async def _send_key(keycode: str) -> str:
    device = await _connect()
    try:
        await device.shell(f"input keyevent {keycode}")
        return f"Sent {keycode}"
    finally:
        await device.close()


mcp = FastMCP("Android TV Remote")


@mcp.tool()
async def power_on() -> str:
    """テレビの電源をオンにする (Wake-on-LAN)"""
    send_magic_packet(TV_MAC)
    return f"WoL packet sent to {TV_MAC}"


@mcp.tool()
async def power_off() -> str:
    """テレビの電源をオフにする"""
    device = await _connect()
    try:
        await device.shell("input keyevent KEYCODE_POWER")
        return "Power off sent"
    finally:
        await device.close()


@mcp.tool()
async def volume_up(steps: int = 1) -> str:
    """音量を上げる

    Args:
        steps: 上げる段階数 (デフォルト: 1)
    """
    device = await _connect()
    try:
        for _ in range(steps):
            await device.shell("input keyevent KEYCODE_VOLUME_UP")
        return f"Volume up x{steps}"
    finally:
        await device.close()


@mcp.tool()
async def volume_down(steps: int = 1) -> str:
    """音量を下げる

    Args:
        steps: 下げる段階数 (デフォルト: 1)
    """
    device = await _connect()
    try:
        for _ in range(steps):
            await device.shell("input keyevent KEYCODE_VOLUME_DOWN")
        return f"Volume down x{steps}"
    finally:
        await device.close()


@mcp.tool()
async def channel(number: int) -> str:
    """チャンネルボタンを押す (1-12)

    Args:
        number: チャンネル番号 (1-12)
    """
    if number not in CHANNEL_KEYCODES:
        return f"Error: channel must be 1-12, got {number}"
    return await _send_key(CHANNEL_KEYCODES[number])


if __name__ == "__main__":
    mcp.run(transport="stdio")
