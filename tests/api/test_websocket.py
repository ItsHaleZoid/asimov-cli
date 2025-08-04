#!/usr/bin/env python3
import asyncio
import websockets
import json


async def test_websocket():
    job_id = "acd6fbd8-9480-4de9-9a19-843174c649b7"
    uri = f"ws://localhost:8000/ws/{job_id}"
    
    try:
        print(f"Connecting to {uri}")
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
            
            # Listen for all messages
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    if data.get('type') == 'status_update':
                        print(f"STATUS UPDATE: {data.get('status')} - {data.get('progress')}% - {data.get('message')}")
                    elif data.get('type') == 'log':
                        print(f"LOG: {data.get('message')}")
                    else:
                        print(f"OTHER: {data}")
                except asyncio.TimeoutError:
                    print("No message received, continuing...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    break
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())