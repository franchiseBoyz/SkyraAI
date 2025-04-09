import requests

api_url = "https://api.aifoundry.com/agents/your-agent-id"
headers = {
    'Authorization': 'Bearer YOUR_API_KEY', 
    'Content-Type': 'application/json',
}
data = {
    'input_data': 'your input to the agent here'
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    print("Response from AI Agent:", response.json())
else:
    print("Error:", response.status_code, response.text)
