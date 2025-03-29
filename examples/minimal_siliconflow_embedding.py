import asyncio
import numpy as np
import aiohttp
import json

# SiliconFlow API设置
SILICONFLOW_API_KEY = "sk-jziyxiaqoxcyjbykymuoqkiehvtxdrvxwfzwsbjaorzjuksv"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 使用高级嵌入模型

async def get_embedding(text: str):
    """使用SiliconFlow API获取文本嵌入"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
        "encoding_format": "float"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(SILICONFLOW_API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                embedding = data["data"][0]["embedding"]
                # 只打印前5个和最后5个元素
                print(f"嵌入向量维度: {len(embedding)}")
                print(f"前5个元素: {embedding[:5]}")
                print(f"后5个元素: {embedding[-5:]}")
                return embedding
            else:
                error_text = await response.text()
                print(f"错误: {error_text}")
                return None

async def main():
    """测试获取文本嵌入"""
    texts = [
        "人工智能正在改变我们的生活方式。",
        "机器学习是人工智能的一个重要分支。",
        "深度学习使机器能够理解图像和语言。"
    ]
    
    print("测试SiliconFlow嵌入模型(BAAI/bge-m3)...")
    for text in texts:
        print(f"\n文本: {text}")
        embedding = await get_embedding(text)
        if embedding:
            # 计算嵌入向量的范数(长度)
            norm = np.linalg.norm(embedding)
            print(f"嵌入向量范数: {norm}")
    
    # 测试相似度
    print("\n计算文本相似度...")
    embeddings = []
    for text in texts:
        embedding = await get_embedding(text)
        if embedding:
            embeddings.append(embedding)
    
    if len(embeddings) == 3:
        # 计算余弦相似度
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
        sim_2_3 = cosine_similarity(embeddings[1], embeddings[2])
        
        print(f"文本1与文本2的相似度: {sim_1_2}")
        print(f"文本1与文本3的相似度: {sim_1_3}")
        print(f"文本2与文本3的相似度: {sim_2_3}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main()) 