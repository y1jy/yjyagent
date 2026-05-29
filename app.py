import os
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = FastAPI(title="通义千问 API 服务", description="个人知识库问答 Agent 后端")

class Question(BaseModel):
    question: str

def call_qwen(prompt: str) -> str:
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    url = f"{base_url}/chat/completions"
    
    API_KEY = os.getenv("QWEN_API_KEY")
    if not API_KEY:
        return "错误：未设置 QWEN_API_KEY 环境变量"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        reply = result['choices'][0]['message']['content']
        return reply
    except requests.exceptions.Timeout:
        return "错误：请求超时"
    except requests.exceptions.ConnectionError:
        return "错误：网络连接失败"
    except requests.exceptions.HTTPError as e:
        return f"错误：HTTP错误 {e}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"错误：解析失败 {e}"
    except Exception as e:
        return f"错误：{str(e)}"

@app.get("/")
def root():
    return {"service": "通义千问 API 服务", "status": "running"}

@app.post("/chat")
def chat(question: Question):
    if not question.question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    answer = call_qwen(question.question)
    return {"question": question.question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
