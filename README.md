# Sheria Kiganjani AI

An intelligent legal assistant specialized for East African legal services, providing bilingual support in English and Swahili.

## Features

- Bilingual legal assistance (English/Swahili)
- Legal document processing and analysis
- Secure document handling
- Cultural-sensitive responses
- Integration with Claude AI

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Claude API key:
   ```
   CLAUDE_API_KEY=your_api_key_here
   ```

## Running Tests

```bash
pytest tests/
```

## Project Structure

```
sheria-kiganjani/
├── app/
│   ├── __init__.py
│   ├── core/
│   ├── api/
│   ├── models/
│   └── services/
├── tests/
├── .env
├── requirements.txt
└── README.md
```

## Security

- End-to-end encryption
- Secure document handling
- GDPR and data protection compliance
- Role-based access control

## License

MIT License
