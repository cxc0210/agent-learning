import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.9,
)


@tool
def read_file(file_path: Annotated[str, "文件路径"]) -> str:
    """读取指定路径的文件内容，并返回内容字符串。"""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    content_length = len(content.encode("utf-8"))
    print(f"[工具调用] 读取文件内容: {path}，内容长度: {content_length}个字节")
    return f"文件内容: {content}"


tools = [read_file]
tools_by_name = {current_tool.name: current_tool for current_tool in tools}
model_with_tools = model.bind_tools(tools, tool_choice="auto")

messages = [
    SystemMessage(
        content=f"""你是一个助手，可以调用工具读取文件内容。
工作流程：
1. 用户要求读取文件内容时，调用read_file工具读取文件内容。
2. 将读取到的文件内容返回给用户。
3. 如果用户要求你介绍自己，需要说明你是一个可以调用工具的助手，并且当前使用的模型是{os.getenv("OPENAI_MODEL_NAME")}。"""
    ),
    HumanMessage(content="请使用read_file工具读取../.gitignore，并告诉我文件内容"),
]

response = model_with_tools.invoke(messages)
max_tool_rounds = 5
tool_round = 0

while response.tool_calls:
    if tool_round >= max_tool_rounds:
        raise RuntimeError(f"工具调用超过最大轮数：{max_tool_rounds}")

    tool_round += 1
    print(f"\n检测到 {len(response.tool_calls)} 个工具调用，开始处理...")

    # ToolMessage 前必须先加入包含 tool_calls 的 AIMessage。
    messages.append(response)

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)

        if selected_tool is None:
            tool_message = ToolMessage(
                content=f"[工具调用失败] 未找到工具：{tool_name}",
                tool_call_id=tool_call["id"],
            )
        else:
            print(f"[工具调用] {tool_name}，参数：{tool_call['args']}")

            try:
                # 传入完整 ToolCall，LangChain 会自动生成关联的 ToolMessage。
                tool_message = selected_tool.invoke(tool_call)
            except (OSError, UnicodeError) as error:
                tool_message = ToolMessage(
                    content=f"[工具调用失败] {error}",
                    tool_call_id=tool_call["id"],
                )

        messages.append(tool_message)

    response = model_with_tools.invoke(messages)

print(response.content)

