
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


# Load config

with open(PROJECT_DIR / "config.json", "r") as f:
    config = json.load(f)
provider = config["provider"]


# Initialize the LLM
def get_llm():
    if provider=="openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to the project's .env file "
                "or set it as an environment variable."
            )
        llm = ChatOpenAI(
            model = config["openai"]["model"],
            temperature = config["openai"]["temperature"],
            max_tokens = config["openai"]["max_token"],
            api_key=api_key
        )
        return llm
    elif provider=="gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add it to the project's .env file "
                "or set it as an environment variable."
            )
        llm = ChatGoogleGenerativeAI(
            model = config["gemini"]["model"],
            temperature = config["gemini"]["temperature"],
            max_tokens = config["gemini"]["max_token"],
            api_key=api_key
        )
        return llm
    else:
        raise ValueError("Invalid provider in config.json [It must be openai or gemini only]")