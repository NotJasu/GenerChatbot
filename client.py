import requests

# Main loop to prompt for questions
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    # Send the question to the Flask app via a POST request
    payload = {'query': question}
    response = requests.post('http://localhost:8080/api/chat', json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f'Question: {data["query"]}')
        print(f'Answer: {data["answer"]}')
    else:
        print('Error:', response.status_code, response.text)
