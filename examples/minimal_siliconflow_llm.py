import asyncio
import aiohttp
import json

# SiliconFlow API设置
SILICONFLOW_API_KEY = "sk-jziyxiaqoxcyjbykymuoqkiehvtxdrvxwfzwsbjaorzjuksv" 
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_MODEL = "deepseek-ai/DeepSeek-R1"  # 使用DeepSeek-R1模型

async def get_llm_response(prompt, system_prompt=None):
    """使用SiliconFlow API获取LLM响应"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False
    }
    
    print(f"发送请求到SiliconFlow API (模型: {LLM_MODEL})...")
    async with aiohttp.ClientSession() as session:
        async with session.post(SILICONFLOW_API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"\n响应状态: {response.status}")
                print(f"Token使用情况: {result['usage']}")
                return content
            else:
                error_text = await response.text()
                print(f"错误: {error_text}")
                return None

async def main():
    """测试获取LLM响应"""
    # 测试普通用户提问
    user_prompt = "简要介绍一下GraphRAG技术是什么？"
    print(f"\n用户提问: {user_prompt}")
    
    response = await get_llm_response(user_prompt)
    if response:
        print(f"\n回答:\n{response}")
    
    # 测试带系统提示的用户提问
    system_prompt = "你是一位人工智能领域的专家，擅长用通俗易懂的语言解释复杂的技术概念。"
    user_prompt = "请解释知识图谱和大语言模型如何结合使用，并给出一些应用场景。"
    
    print(f"\n\n系统提示: {system_prompt}")
    print(f"用户提问: {user_prompt}")
    
    response = await get_llm_response(user_prompt, system_prompt)
    if response:
        print(f"\n回答:\n{response}")
        
    # 测试代码生成能力
    code_prompt = "请用Python编写一个简单的向量搜索算法，实现余弦相似度计算和KNN检索"
    
    print(f"\n\n用户提问: {code_prompt}")
    response = await get_llm_response(code_prompt)
    if response:
        print(f"\n回答:\n{response}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main()) 