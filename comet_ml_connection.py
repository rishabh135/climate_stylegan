from websocket import create_connection

ws = create_connection("wss://www.comet.ml/ws/logger-ws")

print("Sending 'Hello, World'...")

ws.send("Hello, World")

print("Sent")

print("Receiving...")

result =  ws.recv()

print("Received '%s'" % result)

ws.close()