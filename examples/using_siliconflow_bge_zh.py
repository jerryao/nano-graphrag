import sys
import os
import logging
import asyncio
import numpy as np
import aiohttp
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

sys.path.append("..")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nano-graphrag-zh")

WORKING_DIR = "./nano_graphrag_zh_embedding"

# SiliconFlow API设置
SILICONFLOW_API_KEY = "sk-jziyxiaqoxcyjbykymuoqkiehvtxdrvxwfzwsbjaorzjuksv"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # 专为中文优化的嵌入模型
EMBEDDING_MODEL_DIM = 1024  # BGE中文模型的嵌入维度
EMBEDDING_MODEL_MAX_TOKENS = 512  # 此模型的最大token限制


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def siliconflow_zh_embedding(texts: list[str]) -> np.ndarray:
    """
    使用SiliconFlow的BGE中文嵌入模型获取文本嵌入
    """
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    embeddings = []
    
    async with aiohttp.ClientSession() as session:
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
                error_text = await response.text()
                logger.error(f"Error with SiliconFlow API: {error_text}")
                embeddings.append(np.zeros(EMBEDDING_MODEL_DIM))
    
    return np.array(embeddings)


async def main():
    """
    主函数：演示使用SiliconFlow中文嵌入模型
    """
    # 创建工作目录
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # 创建GraphRAG实例
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=siliconflow_zh_embedding,
        # 因为BGE中文模型token限制较小，使用较小的块大小
        chunk_token_size=400,
        chunk_overlap_token_size=50,
    )
    
    # 准备测试文本
    text = """
    人工智能（AI）技术正在快速发展，影响着各行各业。
    
    机器学习作为AI的核心技术，通过海量数据训练模型，使计算机能够像人类一样学习和决策。深度学习是机器学习的一个重要分支，通过多层神经网络模拟人脑的工作方式，实现了图像识别、语音识别和自然语言处理等领域的重大突破。
    
    自然语言处理（NLP）技术使计算机能够理解、解释和生成人类语言，推动了机器翻译、智能客服和信息检索等应用的发展。计算机视觉让机器能够"看到"世界，广泛应用于人脸识别、自动驾驶和医学影像分析等领域。
    
    强化学习通过奖励机制使AI系统在不断尝试中优化行为，已经在游戏、机器人控制和资源调度等领域取得成功。知识图谱整合结构化的知识，为智能搜索、推荐系统和智能问答提供支持。
    
    AI技术的广泛应用也带来了关于隐私、安全和伦理的讨论。如何确保AI系统的公平性、透明度和可解释性，避免偏见和歧视，成为研究者和政策制定者关注的重点。
    
    未来，AI将继续与其他技术融合，推动智能医疗、智慧城市和智能制造等领域的创新发展，为人类社会带来更多可能性。
    """
    
    # 插入文本
    logger.info("正在插入文本...")
    await rag.ainsert(text)
    logger.info("文本插入完成")
    
    # 准备查询
    queries = [
        "人工智能的主要技术有哪些？",
        "AI技术面临哪些挑战？",
        "机器学习和深度学习有什么关系？",
        "AI技术的应用领域有哪些？"
    ]
    
    # 执行查询
    for query in queries:
        logger.info(f"\n查询: {query}")
        
        # 本地模式查询
        result = await rag.aquery(
            query,
            param=QueryParam(
                mode="local",
                response_type="简洁的中文回答",
            )
        )
        
        print(f"回答: {result}")
        print("-" * 50)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main()) 