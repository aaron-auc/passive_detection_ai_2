import requests
import json

def send_post_request_json():
    # API endpoint
    url = "https://pads-website.onrender.com/signals"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Data to be sent
    payload = {
        "confidence": 60
    }
    
    # Send POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    # Check if the request was successful
    if response.status_code == 200 or response.status_code == 201:
        print("POST request successful!")
        print("Response Status Code:", response.status_code)
        print("Response Content:", response.json())
    else:
        print("POST request failed with status code:", response.status_code)
        print("Response:", response.text)

if __name__ == "__main__":
    # Example 1: Send POST request with JSON data
    send_post_request_json()
