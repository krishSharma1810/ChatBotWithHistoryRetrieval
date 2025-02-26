# ChatBot with History Retrieval

## Overview
The ChatBot with History Retrieval is an AI-powered chatbot designed to maintain conversational context by retrieving past interactions. It enhances user experience by providing relevant responses based on historical conversations.

## Features
- Maintains and retrieves past conversation history.
- AI-powered chatbot for interactive and contextual discussions.
- Supports multi-turn conversations for better engagement.
- Streamlit-based user-friendly interface.
- Stores and manages chat history efficiently.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Virtual environment (optional but recommended)

### Clone the Repository
```sh
git clone https://github.com/krishSharma1810/ChatBotWithHistoryRetrieval.git
cd ChatBotWithHistoryRetrieval
```

### Backend Setup
1. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit UI.
2. Start chatting with the AI chatbot.
3. The chatbot retrieves previous interactions to maintain conversation flow.

## Technologies Used
- Python (FastAPI/Flask)
- Streamlit (Frontend UI)
- NLP Libraries (spaCy, Transformers, OpenAI API)
- SQLite or NoSQL database for storing chat history

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit changes and push to your fork.
4. Submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For queries, contact: [sharmakrish1810work@gmail.com](mailto:sharmakrish1810work@gmail.com)
