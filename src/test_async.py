import asyncio

async def get_meow():
    await asyncio.sleep(2)  # Simulating an asynchronous operation
    return 'meow'

async def main():
    response = await get_meow()
    print("meeeeew")
    print(response)

asyncio.run(main())
