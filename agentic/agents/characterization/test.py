"""Minimal Argo model-access check."""

from pathlib import Path
import tomllib

from openai import OpenAI


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"

with CONFIG_PATH.open("rb") as stream:
    config = tomllib.load(stream)["openai"]

client = OpenAI(
    api_key=config["api_key"],
    base_url=config["base_url"],
)

response = client.chat.completions.create(
    model=config["model"],
    messages=[
        {
            "role": "user",
            "content": "What is the stock price for Tesla now?",
        }
    ],
)

print(response.choices[0].message.content)
