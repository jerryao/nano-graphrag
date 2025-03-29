import sys
import os
import json
import logging
import asyncio
import numpy as np
import aiohttp
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

sys.path.append("..")

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "./nano_graphrag_cache_siliconflow_embedding"

# SiliconFlow API设置
SILICONFLOW_API_KEY = "sk-jziyxiaqoxcyjbykymuoqkiehvtxdrvxwfzwsbjaorzjuksv"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 使用高级嵌入模型
EMBEDDING_MODEL_DIM = 1024  # BGE-M3 的嵌入维度为1024
EMBEDDING_MODEL_MAX_TOKENS = 8192  # BGE-M3 支持最大8192个token


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def siliconflow_embedding(texts: list[str]) -> np.ndarray:
    """
    使用SiliconFlow API获取文本嵌入
    """
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    embeddings = []
    
    async with aiohttp.ClientSession() as session:
        # 对每个文本批量处理
        tasks = []
        for text in texts:
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text,
                "encoding_format": "float"
            }
            tasks.append(
                session.post(SILICONFLOW_API_URL, headers=headers, json=payload)
            )
        
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            if response.status == 200:
                data = await response.json()
                embeddings.append(data["data"][0]["embedding"])
            else:
                # 处理错误情况
                error_text = await response.text()
                print(f"Error with SiliconFlow API: {error_text}")
                # 返回零向量作为fallback
                embeddings.append(np.zeros(EMBEDDING_MODEL_DIM))
    
    return np.array(embeddings)


def insert_data():
    """插入测试数据"""
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=siliconflow_embedding,
    )
    
    # 使用示例数据
    with open("../tests/mock_data.txt", encoding="utf-8-sig") as f:
        text = f.read()
    
    print("正在插入数据...")
    rag.insert(text)
    print("数据插入完成！")


def query_data():
    """查询数据"""
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=siliconflow_embedding,
    )
    
    # 本地查询模式
    print("\n---本地查询模式---")
    result_local = rag.query(
        "这个故事的主要主题是什么？", 
        param=QueryParam(mode="local")
    )
    print(result_local)
    
    # 全局查询模式
    print("\n---全局查询模式---")
    result_global = rag.query(
        "这个故事的主要主题是什么？",
        param=QueryParam(mode="global")
    )
    print(result_global)


if __name__ == "__main__":
    # 如果目录不存在，插入数据；否则直接查询
    if not os.path.exists(WORKING_DIR):
        insert_data()
    
    query_data() 