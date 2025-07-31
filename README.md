# Claude Code MLX Proxy

A FastAPI-based proxy server that uses Apple's MLX framework via `mlx_lm` to provide a local, Claude API-compatible interface. This allows you to run large language models locally on Apple Silicon Macs with the same API interface as the official Claude API.

## Features

- **Claude API Compatibility**: Implements the `/v1/chat/completions` endpoint, mirroring the Claude API structure.
- **Local Inference**: Runs large language models locally on your machine using Apple's MLX framework for optimized performance on Apple Silicon.
- **Streaming Support**: Supports both standard and streaming responses.
- **Configurable**: Easily configure the server, model, and generation parameters via a `.env` file.
- **Health Checks**: Includes a `/health` endpoint to monitor server and model status.

## Requirements

- macOS with Apple Silicon (M1, M2, M3, M4 series)
- Python 3.13+
- At least 8GB of RAM (16GB or more is recommended for larger models).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chand1012/claude-code-mlx-proxy.git
    cd claude-code-mlx-proxy
    ```

2.  **Set up the environment:**
    Create a `.env` file from the example. This file will control all the configuration for the proxy.
    ```bash
    cp .env.example .env
    ```
    Now, you can edit the `.env` file to change the model, port, or other settings.

3.  **Install dependencies:**
    This project uses `uv` for package management.
    ```bash
    uv sync
    ```

## Usage

### Starting the Server

Launch the server by running:
```bash
python main.py
```
The server will start on `http://localhost:8000` (or as configured in your `.env` file) and begin loading the specified MLX model.

### Configuration

All settings are managed through the `.env` file.

| Variable              | Default                                       | Description                                                                                             |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `HOST`                | `0.0.0.0`                                     | The host address for the server.                                                                        |
| `PORT`                | `8000`                                        | The port for the server.                                                                                |
| `MODEL_NAME`          | `mlx-community/GLM-4.5-Air-3bit`              | The MLX model to load from Hugging Face. Find more at the [MLX Community](https://huggingface.co/mlx-community). |
| `API_MODEL_NAME`      | `claude-4-sonnet-20250514`                    | The model name that will be returned in the API responses. You can set this to match a known Claude model. |
| `TRUST_REMOTE_CODE`   | `false`                                       | Set to `true` if the model tokenizer requires trusting remote code.                                     |
| `EOS_TOKEN`           | `None`                                        | The End-of-Sequence token, required for some models like Qwen. Example: `"<|endoftext|>"`                 |
| `DEFAULT_MAX_TOKENS`  | `4096`                                        | The default maximum number of tokens to generate in a response.                                         |
| `DEFAULT_TEMPERATURE` | `1.0`                                         | The default temperature for generation (creativity).                                                    |
| `DEFAULT_TOP_P`       | `1.0`                                         | The default top-p for generation.                                                                       |
| `VERBOSE`             | `false`                                       | Set to `true` to enable verbose logging from the MLX generate function.                                 |

## API Usage

The proxy is designed to be a drop-in replacement for clients using the Claude API.

### Example with `curl`

**Non-streaming:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "messages": [
      {"role": "user", "content": "Explain what MLX is in three sentences."}
    ],
    "max_tokens": 150
  }'
```

**Streaming:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "messages": [
      {"role": "user", "content": "Write a short story about a robot who discovers music."}
    ],
    "stream": true,
    "max_tokens": 500
  }'
```

### Python Client Example

```python
import requests
import json

def get_local_claude_completion(prompt: str, stream: bool = False):
    """
    A simple client to test the local proxy.
    """
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "claude-4-sonnet-20250514",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 1024,
    }
    
    with requests.post(url, json=payload, stream=stream) as response:
        if response.ok:
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_part = line_str[6:]
                            if data_part == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data_part)
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                print(content, end='', flush=True)
                                full_response += content
                            except json.JSONDecodeError:
                                pass # Ignore empty or malformed lines
                return full_response
            else:
                data = response.json()
                return data['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

# Example usage
print("--- Non-Streaming ---")
non_streaming_response = get_local_claude_completion("What is the capital of France?")
print(non_streaming_response)

print("\n\n--- Streaming ---")
get_local_claude_completion("Write a haiku about Python programming.", stream=True)
print()
```

## API Endpoints

-   `POST /v1/chat/completions`: The main endpoint for chat completions.
-   `GET /health`: A health check endpoint that returns the server status and whether the model is loaded.
-   `GET /`: A root endpoint with basic status information.

## License

This project is licensed under the MIT License.
