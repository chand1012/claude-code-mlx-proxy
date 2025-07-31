import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mlx_lm import load, generate, stream_generate
from config import config

# Global variables for model and tokenizer
model = None
tokenizer = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = config.API_MODEL_NAME
    messages: List[Message]
    max_tokens: Optional[int] = config.DEFAULT_MAX_TOKENS
    temperature: Optional[float] = config.DEFAULT_TEMPERATURE
    top_p: Optional[float] = config.DEFAULT_TOP_P
    stream: Optional[bool] = False
    system: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer
    print(f"Loading MLX model: {config.MODEL_NAME}")

    # Prepare tokenizer config
    tokenizer_config = {}
    if config.TRUST_REMOTE_CODE:
        tokenizer_config["trust_remote_code"] = True
    if config.EOS_TOKEN:
        tokenizer_config["eos_token"] = config.EOS_TOKEN

    model, tokenizer = load(config.MODEL_NAME, tokenizer_config=tokenizer_config)
    print("Model loaded successfully!")
    yield
    # Cleanup on shutdown
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


def format_messages_for_llama(
    messages: List[Message], system: Optional[str] = None
) -> str:
    """Convert Claude-style messages to Llama format"""
    formatted_messages = []

    # Add system message if provided
    if system:
        formatted_messages.append({"role": "system", "content": system})

    # Add user messages
    for message in messages:
        formatted_messages.append({"role": message.role, "content": message.content})

    # Apply chat template if available
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            formatted_messages, add_generation_prompt=True
        )
    else:
        # Fallback formatting
        prompt = ""
        for msg in formatted_messages:
            if msg["role"] == "system":
                prompt += f"<|system|>\n{msg['content']}\n<|end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n<|end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n<|end|>\n"
        prompt += "<|assistant|>\n"
        return prompt


def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format messages for Llama
        prompt = format_messages_for_llama(request.messages, request.system)

        # Count input tokens
        input_tokens = count_tokens(prompt)

        if request.stream:
            return StreamingResponse(
                stream_generate_response(request, prompt, input_tokens),
                media_type="text/plain",
            )
        else:
            return await generate_response(request, prompt, input_tokens)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_response(
    request: ChatCompletionRequest, prompt: str, input_tokens: int
):
    """Generate non-streaming response"""
    # Generate text
    response_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        verbose=config.VERBOSE,
    )

    # Count output tokens
    output_tokens = count_tokens(response_text)

    # Create response
    response = ChatCompletionResponse(
        id="chatcmpl-" + str(hash(prompt))[:8],
        created=int(asyncio.get_event_loop().time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )

    return response


async def stream_generate_response(
    request: ChatCompletionResponse, prompt: str, input_tokens: int
):
    """Generate streaming response"""
    response_id = "chatcmpl-" + str(hash(prompt))[:8]
    created = int(asyncio.get_event_loop().time())

    # Stream generation
    for i, response in enumerate(
        stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            verbose=config.VERBOSE,
        )
    ):
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant" if i == 0 else None,
                        "content": response.text,
                    },
                    "finish_reason": None,
                }
            ],
        )

        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk
    final_chunk = ChatCompletionChunk(
        id=response_id,
        created=created,
        model=request.model,
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
    )

    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    print(f"Starting Claude Code MLX Proxy on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
