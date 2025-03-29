import sys
import os
import json
import logging
import asyncio
import numpy as np
import aiohttp
from typing import List, Union, Dict, Any, Optional
from dataclasses import dataclass
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs, compute_args_hash
from nano_graphrag.base import BaseKVStorage

sys.path.append("..")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("siliconflow-graphrag")

# 工作目录
WORKING_DIR = "./siliconflow_graphrag"

# SiliconFlow API设置
SILICONFLOW_API_KEY = "sk-jziyxiaqoxcyjbykymuoqkiehvtxdrvxwfzwsbjaorzjuksv"
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# 嵌入模型设置
EMBEDDING_API_URL = f"{SILICONFLOW_API_BASE}/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL_DIM = 1024
EMBEDDING_MODEL_MAX_TOKENS = 8192

# LLM模型设置
LLM_API_URL = f"{SILICONFLOW_API_BASE}/chat/completions"
LLM_MODEL = "deepseek-ai/DeepSeek-R1"  # 使用DeepSeek-R1模型
LLM_MAX_TOKENS = 4096


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def siliconflow_embedding(texts: List[str]) -> np.ndarray:
    """
    使用SiliconFlow API获取文本嵌入
    """
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    embeddings = []
    
    # 对超长文本进行特殊处理
    processed_texts = []
    for text in texts:
        # 简单截断过长文本
        if len(text) > 6000:  # 估计约8192个token的限制
            text = text[:6000]
        processed_texts.append(text)
    
    async with aiohttp.ClientSession() as session:
        # 使用gather实现并行请求
        tasks = []
        for text in processed_texts:
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text,
                "encoding_format": "float"
            }
            tasks.append(
                session.post(EMBEDDING_API_URL, headers=headers, json=payload)
            )
        
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            if response.status == 200:
                data = await response.json()
                embeddings.append(data["data"][0]["embedding"])
            else:
                error_text = await response.text()
                logger.error(f"Error with SiliconFlow embedding API: {error_text}")
                # 返回零向量作为fallback
                embeddings.append(np.zeros(EMBEDDING_MODEL_DIM))
    
    return np.array(embeddings)


async def siliconflow_complete(
    prompt: str, 
    system_prompt: Optional[str] = None, 
    history_messages: List[Dict[str, str]] = [],
    max_tokens: int = 1024,
    hashing_kv: Optional[BaseKVStorage] = None,
    **kwargs
) -> str:
    """
    使用SiliconFlow API的DeepSeek-R1模型完成文本生成
    """
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # 添加历史消息
    messages.extend(history_messages)
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": prompt})

    # 如果有缓存，尝试从缓存获取
    if hashing_kv is not None:
        args_hash = compute_args_hash(LLM_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,  # 使用较低的温度以获得确定性结果
        "top_p": 0.95,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 如果有缓存，保存到缓存
                if hashing_kv is not None:
                    await hashing_kv.upsert(
                        {args_hash: {"return": content, "model": LLM_MODEL}}
                    )
                    await hashing_kv.index_done_callback()
                
                return content
            else:
                error_text = await response.text()
                logger.error(f"Error with SiliconFlow LLM API: {error_text}")
                return "发生错误，无法生成回复。"


async def main():
    """
    主函数
    """
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # 创建GraphRAG实例，使用我们自定义的SiliconFlow函数
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=siliconflow_embedding,
        best_model_func=siliconflow_complete,  # 使用DeepSeek-R1作为主要LLM
        cheap_model_func=siliconflow_complete,  # 同样使用DeepSeek-R1作为次要LLM
        chunk_token_size=800,  # 适当调整块大小
        chunk_overlap_token_size=100,
    )
    
    # 准备测试文本
    sample_text = """
    人工智能（AI）技术正在快速发展，影响着各行各业。
    
    机器学习作为AI的核心技术，通过海量数据训练模型，使计算机能够像人类一样学习和决策。深度学习是机器学习的一个重要分支，通过多层神经网络模拟人脑的工作方式，实现了图像识别、语音识别和自然语言处理等领域的重大突破。
    
    自然语言处理（NLP）技术使计算机能够理解、解释和生成人类语言，推动了机器翻译、智能客服和信息检索等应用的发展。计算机视觉让机器能够"看到"世界，广泛应用于人脸识别、自动驾驶和医学影像分析等领域。
    
    强化学习通过奖励机制使AI系统在不断尝试中优化行为，已经在游戏、机器人控制和资源调度等领域取得成功。知识图谱整合结构化的知识，为智能搜索、推荐系统和智能问答提供支持。
    
    AI技术的广泛应用也带来了关于隐私、安全和伦理的讨论。如何确保AI系统的公平性、透明度和可解释性，避免偏见和歧视，成为研究者和政策制定者关注的重点。
    
    未来，AI将继续与其他技术融合，推动智能医疗、智慧城市和智能制造等领域的创新发展，为人类社会带来更多可能性。
    """
    
    # 如果工作目录中没有已处理的数据
    if not os.path.exists(os.path.join(WORKING_DIR, "chunk_entity_relation")):
        # 插入文本
        logger.info("正在插入文本...")
        await rag.ainsert(sample_text)
        logger.info("文本插入完成")
    
    # 准备查询
    queries = [
        "人工智能的主要技术有哪些？",
        "AI技术面临哪些挑战？",
        "机器学习和深度学习有什么关系？",
    ]
    
    # 执行查询
    for query in queries:
        logger.info(f"\n查询: {query}")
        
        # 本地模式查询
        logger.info("本地模式查询结果:")
        result_local = await rag.aquery(
            query,
            param=QueryParam(
                mode="local",
                response_type="详细的中文回答",
            )
        )
        print(result_local)
        print("\n" + "-"*50 + "\n")
        
        # 全局模式查询
        logger.info("全局模式查询结果:")
        result_global = await rag.aquery(
            query,
            param=QueryParam(
                mode="global",
                response_type="详细的中文回答",
            )
        )
        print(result_global)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main()) 