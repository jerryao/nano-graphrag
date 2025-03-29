import os
import asyncio
import logging
import aiohttp
from nano_graphrag import GraphRAG, QueryParam

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("siliconflow-runner")

# API设置
SILICONFLOW_API_KEY = "sk-jziyxiaqoxcyjbykymuoqkiehvtxdrvxwfzwsbjaorzjuksv"
API_BASE = "https://api.siliconflow.cn/v1"

# 工作目录
WORKING_DIR = "./siliconflow_minimal_test"

# 定义异步函数来执行完整流程
async def run_graphrag_with_siliconflow():
    """
    使用SiliconFlow API运行完整的GraphRAG流程
    """
    # 确保工作目录存在
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # 1. 定义一个函数来获取测试数据 - 简单起见我们直接使用静态数据
    test_text = """人工智能正在迅速发展并改变各行各业。机器学习作为其核心技术，通过大量数据学习模式和规则。
    深度学习是机器学习的一个分支，使用神经网络模拟人脑工作方式。自然语言处理让计算机能理解和生成人类语言。
    计算机视觉则使机器能"看到"并理解图像和视频。这些技术推动了智能客服、自动驾驶、医疗诊断等应用的发展。"""
    
    # 2. 测试SiliconFlow API连接
    logger.info("测试SiliconFlow API连接...")
    
    # 测试嵌入API
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 测试嵌入API
        embed_payload = {
            "model": "BAAI/bge-m3",
            "input": "API测试文本",
            "encoding_format": "float"
        }
        
        logger.info("测试嵌入API...")
        async with session.post(
            f"{API_BASE}/embeddings",
            headers=headers,
            json=embed_payload
        ) as response:
            if response.status == 200:
                logger.info("嵌入API连接成功!")
            else:
                logger.error(f"嵌入API连接失败: {await response.text()}")
                return
    
    # 这里我们可以使用siliconflow_complete_example.py中的代码，但为简化流程，我们直接使用OpenAI（如果已设置）
    # 或者在这里使用一个最小化的GraphRAG配置
    
    # 3. 初始化GraphRAG
    logger.info("初始化GraphRAG...")
    
    # 注意: 这个最小化示例只测试向量嵌入，实体抽取仍然使用OpenAI
    # 如需完整的SiliconFlow实现，请运行siliconflow_complete_example.py
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        # 仅对简单测试插入和查询过程
    )
    
    # 4. 插入文本
    logger.info("插入测试文本...")
    await rag.ainsert(test_text)
    logger.info("文本插入完成")
    
    # 5. 查询
    logger.info("进行查询测试...")
    query = "机器学习和深度学习的关系是什么?"
    
    try:
        result = await rag.aquery(
            query,
            param=QueryParam(mode="naive", response_type="简短回答")
        )
        
        logger.info(f"查询结果: {result}")
    except Exception as e:
        logger.error(f"查询过程中出错: {str(e)}")
    
    logger.info("测试完成!")

# 运行异步主函数
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_graphrag_with_siliconflow()) 