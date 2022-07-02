import asyncio
from bleak import BleakScanner, BleakClient

address =  "3F529E98-B41A-A1BF-2DB7-8CCE37358399"
addr = "19b10001-e8f2-537e-4f6c-d104768a1214"

async def main():
    async with BleakClient(address) as client:
        raw = await client.read_gatt_char(addr)
        print(raw)
        for r in raw:
            print(r)
        val = raw.decode()
        print(val)

asyncio.run(main())

