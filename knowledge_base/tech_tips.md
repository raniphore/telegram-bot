# Tech Tips & Troubleshooting

## Python Virtual Environments
Always use a virtual environment for Python projects to avoid dependency conflicts.
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Git Best Practices
- Commit often with clear messages
- Use feature branches, never commit directly to main
- Always pull before pushing: `git pull origin main`
- Use `.gitignore` to exclude secrets and generated files

## API Keys Security
Never hardcode API keys in source code. Use environment variables and a `.env` file. Add `.env` to `.gitignore`.

## Docker Tips
Use `docker-compose` for multi-service applications. Always specify explicit image versions in `Dockerfile` for reproducibility.

## SQLite Performance
For small datasets (under 100k rows), SQLite is sufficient and requires no server setup. Use WAL mode for better concurrent reads:
```sql
PRAGMA journal_mode=WAL;
```

## Debugging Tips
- Use `logging` module instead of print statements in production
- Set log level to DEBUG during development: `logging.basicConfig(level=logging.DEBUG)`
- Use `try/except` blocks to handle expected failures gracefully
