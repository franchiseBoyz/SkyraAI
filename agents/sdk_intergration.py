import requests

api_url = "https://api.aifoundry.com/agents/agent_id"
headers = {
    'Authorization': 'Bearer YOUR_API_KEY', 
    'Content-Type': 'application/json',
}
data = {
    'input_data': 'Multi agent investment manager'
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    print("Response from AI Agent:", response.json())
else:
    print("Error:", response.status_code, response.text)
