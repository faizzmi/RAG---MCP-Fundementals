# this code will be just a simple example to show how to use the agent to interact with other platform using tool n it will not run 


def fetch_flight_details():
    flights = []

    malaysia_airlines = call_tool("Malaysia Airlines", {origin, destination, date})
    flights.append(malaysia_airlines)

    airasia_airline = call_tool("AirAsia", {origin, destination, date})
    flights.append(airasia_airline)


    return flights

def call_tool(tool_name, args):
    if tool_name == "Malaysia Airlines":
        response = call_api("https://api.malaysiaairlines.com/flights", args)
        return {
            {
                "flightnumber": flight["flightNum"],
                "origin": flight["from"],
                "destination": flight["to"],
            } for flight in response["flight-info"]
        }
    elif tool_name == "Air Asia":
        response = call_api("https://api.malaysiaairlines.com/flights", args)
        return {
            {
                "flightnumber": flight["flightNum"],
                "origin": flight["from"],
                "destination": flight["to"],
            } for flight in response["flight-info"]
        }
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
    

user_input = input("Where do you want to fly?\n")

prompt = [
    {"role": "system", "content": "Extract origin, destination and date"},
    {"role": "user", "content": user_input}
]

flight_query = call_llm(prompt)

# agent interact to other platform using tool for each 
flight_options = fetch_flight_details()

user_prefs = fetch_user_preferences()

decision_prompt = [
    {"role": "system", "content": f"User preferences: {user_prefs}"},
    {"role": "user", "content": f"Available flights: {flight_options}"}
]

desicion = call_llm(decision_prompt)

def book_flight(flight): return f"Flight {flight['flight']} on {flight['airline']} booked successfully!"

# assume the output like this first
print(book_flight({"flight": "MH123", "airline": "Malaysia Airlines"}))


# uri should define as uri