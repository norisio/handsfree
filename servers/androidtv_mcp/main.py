import asyncio
from pathlib import Path

from adb_shell.adb_device_async import AdbDeviceTcpAsync
from adb_shell.auth.keygen import keygen
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

KEY_PATH = Path(__file__).parent / "adbkey"


async def main():
    # 鍵がなければ生成
    if not KEY_PATH.exists():
        print("Generating ADB key pair...")
        keygen(str(KEY_PATH))

    with open(KEY_PATH) as f:
        priv = f.read()
    with open(str(KEY_PATH) + ".pub") as f:
        pub = f.read()

    signer = PythonRSASigner(pub, priv)

    device = AdbDeviceTcpAsync("192.168.10.111", 5555)
    print("Connecting... (テレビに許可ダイアログが出たら許可してください)")
    await device.connect(rsa_keys=[signer], auth_timeout_s=30)
    print("Connected!")

    # 接続テスト: デバイス情報を取得
    response = await device.shell("getprop ro.product.model")
    print(f"Model: {response.strip()}")

    response = await device.shell("getprop ro.build.display.id")
    print(f"Build: {response.strip()}")

    await device.close()


if __name__ == "__main__":
    asyncio.run(main())
