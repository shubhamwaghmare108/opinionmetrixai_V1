# OpinionMetrix AI

A Streamlit application for three-class sentiment analysis of customer reviews using a fine-tuned Transformer model.

## Features

- Positive, neutral, and negative sentiment prediction
- Probability breakdown for each class
- Recent prediction history within the current session
- Optional MySQL persistence
- Automatic model-artifact download from Google Drive

## Local setup

Use Python 3.10 or 3.11 for the most predictable compatibility with the ML stack.

```bash
git clone https://github.com/shubhamwaghmare108/opinionmetrixai_V1.git
cd opinionmetrixai_V1
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Model artifacts

The application expects the downloaded model at:

```text
models/
└── transformer/
    ├── config.json
    ├── label_encoder.pkl
    ├── tokenizer.json or tokenizer files
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── model weights such as model.safetensors or pytorch_model.bin
```

The Google Drive folder configured in `download_models.py` must produce this directory structure. Do not commit model weights or secrets to Git.

## Streamlit Cloud deployment

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select this repository, the `main` branch, and `app.py` as the entrypoint.
4. Add the required secrets under **Advanced settings → Secrets**.
5. Deploy and inspect the application logs if model downloading fails.

The app can run without MySQL; predictions will still be generated, but they will not be persisted.

## Optional MySQL secrets

Add this TOML to Streamlit secrets only if persistence is required:

```toml
[connections.mysql]
host = "your-mysql-host"
port = 3306
username = "your-username"
password = "your-password"
database = "your-database"
# Optional CA certificate content for TLS connections
# ssl_ca = "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----"
```

The target database must contain a `Prediction` table with these columns:

- `review`
- `sentiment`
- `prob_positive`
- `prob_neutral`
- `prob_negative`
- `correct_sentiment`

## Development checks

Before deploying, run:

```bash
python -m compileall app.py db.py download_models.py src
```

For production changes, add and run automated tests before merging.

## Security notes

- Keep database credentials in Streamlit secrets or environment-specific secret management.
- Never place credentials in source files, notebooks, or commit history.
- Review Google Drive sharing permissions before deployment.
- Avoid displaying raw infrastructure exceptions to end users.
