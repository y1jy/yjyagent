import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
load_dotenv()

@tool
def calculate(expression:str) -> str:
    """计算数学表达式
     参数: expression (str): 要计算的数学表达式
     返回: str: 计算结果"""
    try:
        allowed_names = {
            'abs': abs,
            'round': round,
            'pow': pow,
            'max': max,
            'min': min,
            "round": round,
            "math": __import__('math'),
        }
        result = eval(expression,{"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

model = ChatOpenAI(
    model="qwen-turbo",
    temperature=0,
    timeout=10,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_tokens=1000
)

system_prompt = """你是一个数学计算助手。

你有以下工具可用：
- calculate(expression): 计算数学表达式

规则：
1. 遇到数学计算问题，必须调用 calculate 工具
2. 不要自己计算，必须用工具
3. 非数学问题直接回答"""



agent = create_agent(
    model=model,
    tools=[calculate],
    system_prompt=system_prompt
)

if __name__ == "__main__":
    test_questions = [
        "12345 * 67890 等于多少？",
        "(100 + 200) * 3 的结果",
        "你好"
    ]
    
    for q in test_questions:
        print("=" * 50)
        print(f"问题: {q}")
        # 关键：用 "messages" 而不是 "input"
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        # 提取最后一条消息的回答
        last_message = result['messages'][-1]
        print(f"回答: {last_message.content}")