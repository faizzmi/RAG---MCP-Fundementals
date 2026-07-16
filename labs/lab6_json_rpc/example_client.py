from jsonrpcclient import request, parse, Ok, Error
import requests

# create jsnrpc request for add funciton
rpc_request = request("add", {"a": 10, "b": 5})

# send HTTP to request thru jsonrpx server
response = requests.post(
    "http://localhost:5000", 
    data=rpc_request, 
    timeout=5
)

# handle the response
if response.status_code == 200:
    result = parse(response.json())
    if isinstance(result, Ok):
        print(f"Result: {result.result}")
    elif isinstance(result, Error):
        print(f"Error: {result.message}")