import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { SystemMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import fs from 'fs/promises';
import 'dotenv/config';

const model = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL_NAME,
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0.9,
    configuration: {
        basePath: process.env.OPENAI_BASE_URL,
    }
});

const readFileTool = tool(
    async ({ filePath }) => {
        const content = await fs.readFile(filePath, 'utf-8');
        console.log(`[工具调用] 读取文件内容: ${filePath}，内容长度: ${content.length}个字节`);
        return `文件内容: ${content}`;
    },
    {
        name: 'read_file',
        description: '读取指定路径的文件内容，返回文件内容字符串',
        schema: z.object({
            filePath: z.string().describe('文件路径'),
        }),
    }
);

const tools = [readFileTool];
const modelWithTools = model.bindTools(tools);

const messages = [
    new SystemMessage(`你是一个助手,可以调用工具读取文件内容。
        工作流程: 
        1. 用户要求读取文件内容, 你调用read_file工具读取文件内容。
        2. 你将读取到的文件内容返回给用户。
        3. 如果用户要求你介绍自己, 你需要说明你是一个可以调用工具的助手, 并且当前使用的模型是${process.env.OPENAI_MODEL_NAME}。
        `),
    new HumanMessage("请使用read_file工具读取../.gitignore，并告诉我文件内容"),

]

let response = await modelWithTools.invoke(messages);
messages.push(response);

while (response.tool_calls && response.tool_calls.length > 0) {
    console.log(`\n检查到 ${response.tool_calls.length} 个工具调用, 开始处理工具调用...`);

    //执行工具调用
    const toolResults = await Promise.all(
        response.tool_calls.map(async (toolCall) => {
            const tool = tools.find((t) => t.name === toolCall.name);
            if (!tool) {
                return '[工具调用失败] 未找到工具: ' + toolCall.name;
            }
            console.log(`[工具调用] 调用工具: ${toolCall.name}，参数: ${JSON.stringify(toolCall.args)}`);

            try {
                const result = await tool.invoke(toolCall.args);
                return result;
            } catch (error) {
                console.error(`[工具调用失败] 调用工具 ${toolCall.name} 时发生错误:`, error);
                return `[工具调用失败] 调用工具 ${toolCall.name} 时发生错误: ${error.message}`;
            }
        })
    );

    response.tool_calls.forEach((toolCall, index) => {
        messages.push(
            new ToolMessage({
                content: toolResults[index],
                tool_call_id: toolCall.id,
            })
        )
    });

    response = await modelWithTools.invoke(messages);
}

console.log(response.content);
