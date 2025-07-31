# Local MLX Backend for Claude Code

This project provides a local server that acts as a backend for the **Claude Code** IDE extension. It allows you to use open-source models running on your local machine via Apple's MLX framework. Instead of sending your code to Anthropic's servers, you can use powerful models like Llama 3, Mistral, DeepSeek, and more, all running on your Apple Silicon Mac.

This server mimics the Anthropic API that Claude Code communicates with, redirecting all requests to a local model of your choice.

## Why Use a Local Backend with Claude Code?

-   **Total Privacy**: Your code, prompts, and conversations never leave your local machine.
-   **Use Any Model**: Experiment with thousands of open-source models from the [MLX Community on Hugging Face](https://huggingface.co/mlx-community).
-   **Work Offline**: Get code completions and chat with your local model without an internet connection.
-   **No API Keys or Costs**: Run powerful models without needing to manage API keys or pay for usage.
-   **Full Customization**: You have complete control over model parameters and generation settings.

## How to Set It Up

There are two parts: running the local server, and configuring Claude Code to use it.

### Part 1: Run the Local Server

First, get the proxy server running on your machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chand1012/claude-code-mlx-proxy.git
    cd claude-code-mlx-proxy
    ```

2.  **Set up the environment:**
    Create a `.env` file from the example. This file controls the configuration for the proxy, including which model to run.
    ```bash
    cp .env.example .env
    ```
    You can edit the `.env` file now or later to change the model or port.

3.  **Install dependencies:**
    This project uses `uv` for fast package management.
    ```bash
    uv sync
    ```

4.  **Start the server:**
    ```bash
    python main.py
    ```
    The server will start on `http://localhost:8000` (or as configured in your `.env`) and begin downloading and loading the specified MLX model. This may take some time on the first run.

### Part 2: Configure Claude Code

Next, tell your Claude Code extension to send requests to your local server instead of the official Anthropic API.

As described in the [official Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway), you do this by setting the `ANTHROPIC_BASE_URL` environment variable.

The most reliable way to do this is to **launch your IDE from a terminal** where the variable has been set:

```bash
# Set the environment variable to point to your local server
export ANTHROPIC_BASE_URL=http://localhost:8000

# Now, launch your IDE from this same terminal window
# For VS Code:
code .

# For other IDEs, use their respective command-line launchers.
```

Once your IDE is running, Claude Code will automatically use your local MLX backend. You can now chat with it or use its code completion features, and all requests will be handled by your local model.

### Testing the Server

Before configuring Claude Code, you can verify the server is working correctly by sending it a `curl` request from your terminal:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4-sonnet-20250514",
    "messages": [
      {"role": "user", "content": "Explain what MLX is in one sentence."}
    ],
    "max_tokens": 100
  }'
```

You should receive a JSON response from your local model.

## Configuration (`.env`)

All server settings are managed through the `.env` file.

| Variable              | Default                                       | Description                                                                                             |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `HOST`                | `0.0.0.0`                                     | The host address for the server.                                                                        |
| `PORT`                | `8000`                                        | The port for the server.                                                                                |
| `MODEL_NAME`          | `mlx-community/GLM-4.5-Air-3bit`              | The MLX model to load from Hugging Face. Find more at the [MLX Community](https://huggingface.co/mlx-community). |
| `API_MODEL_NAME`      | `claude-4-sonnet-20250514`                    | The model name that the API will report. Set this to a known Claude model to ensure client compatibility. |
| `TRUST_REMOTE_CODE`   | `false`                                       | Set to `true` if the model tokenizer requires trusting remote code.                                     |
| `EOS_TOKEN`           | `None`                                        | The End-of-Sequence token, required for some models like Qwen. Example: `"<|endoftext|>"`                 |
| `DEFAULT_MAX_TOKENS`  | `4096`                                        | The default maximum number of tokens to generate in a response.                                         |
| `DEFAULT_TEMPERATURE` | `1.0`                                         | The default temperature for generation (creativity).                                                    |
| `DEFAULT_TOP_P`       | `1.0`                                         | The default top-p for generation.                                                                       |
| `VERBOSE`             | `false`                                       | Set to `true` to enable verbose logging from the MLX generate function.                                 |

## License

This project is licensed under the MIT License.
