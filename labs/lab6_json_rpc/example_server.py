from http import server
from jsonrpcserver import Result, method, Success, Error

@method
def add(a: float, b: float) -> Result:
    return Success(a + b)

if __name__ == "__main__":
    server(port=8080)