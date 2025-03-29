import sys
import os
import logging
import asyncio
import numpy as np
import aiohttp
from dataclasses import dataclass
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from nano_graphrag._storage import NanoVectorDBStorage

sys.path.append("..")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chinese-document-graphrag")

# 工作目录
WORKING_DIR = "./chinese_documents_graphrag"

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
    
    # 对超长文本进行特殊处理
    processed_texts = []
    for text in texts:
        # 简单截断过长文本，实际应用中可能需要更复杂的处理
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
                logger.error(f"Error with SiliconFlow API: {error_text}")
                # 返回零向量作为fallback
                embeddings.append(np.zeros(EMBEDDING_MODEL_DIM))
    
    return np.array(embeddings)


@dataclass
class ChineseDocument:
    """
    中文文档数据结构
    """
    title: str
    content: str
    source: str
    date: str


def prepare_sample_documents() -> list[ChineseDocument]:
    """
    准备示例中文文档
    """
    return [
        ChineseDocument(
            title="人工智能发展历史",
            content="""人工智能（Artificial Intelligence，简称AI）的发展历程可以追溯到20世纪50年代。1956年，在达特茅斯会议上，"人工智能"一词被首次提出，标志着AI作为一个正式的研究领域的诞生。

早期的AI研究主要集中在解决问题和符号处理上，包括Arthur Samuel的跳棋程序和Allen Newell与Herbert Simon的"逻辑理论家"程序。这个时期被称为符号主义AI的黄金时代。

然而，到了70年代至80年代，由于技术限制和过高的期望，AI研究进入了一个被称为"AI冬天"的低迷期。

1980年代末，随着计算能力的提升和机器学习算法的发展，AI研究重新焕发活力。特别是在90年代，统计方法在自然语言处理和计算机视觉领域取得了重大突破。

21世纪初，随着深度学习和神经网络的复兴，AI迎来了革命性的发展。2012年，Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton在ImageNet比赛中使用深度卷积神经网络取得了历史性突破，标志着深度学习时代的到来。

近年来，大型语言模型（LLM）如GPT系列、BERT和LLaMA的出现，使AI在自然语言理解和生成方面取得了前所未有的进展。同时，强化学习、图神经网络等技术也在不断发展，推动AI向更复杂的应用领域拓展。

目前，人工智能已经渗透到我们生活的方方面面，从智能助手、自动驾驶到医疗诊断、科学研究等领域，正在深刻改变人类社会。""",
            source="AI研究报告",
            date="2023-01-15"
        ),
        ChineseDocument(
            title="中国传统文化的现代价值",
            content="""中国传统文化是中华民族五千多年文明发展中孕育的精华，包含了丰富的哲学思想、伦理道德、文学艺术等内容。在现代社会，这些传统文化元素不仅没有失去其价值，反而展现出新的生命力和现代意义。

首先，中国传统哲学思想如儒家的"仁爱"、道家的"天人合一"、佛家的"众生平等"等理念，为当代人类面临的生态危机、道德困境提供了重要的思想资源。例如，道家的自然观对现代生态环保理念有着深刻的启示。

其次，传统伦理道德如"忠、孝、仁、义、礼、智、信"等，对构建和谐社会关系、培养个人品德具有重要意义。这些价值观经过现代转化，可以成为现代公民道德的重要组成部分。

第三，中国传统艺术如书法、绘画、音乐、戏曲等，不仅是文化遗产，也是现代艺术创新的重要源泉。当代艺术家通过对传统艺术的继承与创新，创造出兼具传统韵味和现代气息的艺术作品。

第四，传统医学、科技等领域的智慧，如中医药、古代建筑技术等，为解决现代问题提供了独特的思路和方法。例如，中医的整体观念和辨证施治为现代医学的发展提供了补充和启示。

最后，传统文化作为民族认同和文化自信的重要源泉，在全球化背景下尤为重要。通过传统文化的传承和创新，可以增强文化认同感，促进不同文化间的对话与交流。

总之，中国传统文化不是僵化的古董，而是活的智慧，通过创造性转化和创新性发展，可以焕发出新的时代光彩，为人类文明的进步做出贡献。""",
            source="文化研究期刊",
            date="2023-03-22"
        ),
        ChineseDocument(
            title="数字经济发展趋势",
            content="""数字经济作为一种新的经济形态，正在重塑全球经济格局和竞争版图。它以数据资源为关键要素，以现代信息网络为主要载体，以信息通信技术融合应用、全要素数字化转型为重要推动力，不断催生新产业、新业态、新模式。

第一，产业数字化转型加速。传统产业通过数字技术赋能实现升级改造，制造业智能化、服务业数字化、农业精准化发展趋势明显。工业互联网、智能制造、智慧农业等新模式快速发展，推动产业效率提升和结构优化。

第二，数字产业化规模扩大。以5G、人工智能、大数据、云计算、区块链等为代表的新一代信息技术产业快速发展，数字技术创新步伐加快，新型基础设施建设提速，为数字经济发展奠定坚实基础。

第三，数据要素市场化进程推进。随着数据确权、流通、交易、安全等制度框架的逐步完善，数据作为新型生产要素的价值正在逐步释放。数据交易所、数据银行等新型机构不断涌现，推动数据要素市场化配置。

第四，平台经济发展模式变革。数字平台企业在强化合规经营的同时，更加注重技术创新和产业链协同，平台与实体经济深度融合的趋势明显，产业互联网快速发展，为中小企业数字化转型提供支撑。

第五，数字消费新业态繁荣发展。直播电商、社交电商、智慧零售等新模式创新活跃，在线教育、远程医疗、数字文娱等新消费场景不断丰富，促进消费结构升级和消费模式创新。

第六，数字治理体系不断完善。各国纷纷加强数字经济法律法规和政策标准体系建设，数据安全、个人信息保护、算法监管等成为关注焦点，全球数字经济治理合作不断深化。

展望未来，数字经济将继续保持高速增长态势，成为驱动经济增长的主要引擎，也将对社会结构、就业形态、国际关系等产生深远影响。各国需要把握数字化发展机遇，应对数字化挑战，推动数字经济健康可持续发展。""",
            source="经济观察报",
            date="2023-05-10"
        ),
        ChineseDocument(
            title="环境保护与可持续发展",
            content="""环境保护与可持续发展已成为当今世界面临的重大课题。随着工业化、城市化进程的加速，环境污染、生态破坏、资源短缺等问题日益突出，严重威胁人类的生存和发展。

首先，气候变化是全球面临的最紧迫环境挑战之一。根据联合国政府间气候变化专门委员会(IPCC)的报告，人类活动导致的温室气体排放已使全球平均气温上升约1.1°C，导致极端天气事件增加、海平面上升、生物多样性减少等严重后果。应对气候变化需要全球协同减少温室气体排放，发展清洁能源，推动低碳转型。

其次，生物多样性保护刻不容缓。据统计，全球约有100万种植物和动物物种面临灭绝威胁，生态系统服务功能退化。保护生物多样性不仅关系到生态平衡，也关系到人类福祉和经济发展。建立自然保护区网络、控制外来入侵物种、减少栖息地破坏是保护生物多样性的重要措施。

第三，水资源短缺和水污染问题日益严重。全球约有20亿人无法获得安全饮用水，水污染导致水生态系统退化、水源性疾病传播。推广节水技术、加强污水处理、完善水资源管理体系是解决水问题的关键。

第四，城市环境问题复杂多样。随着城市化进程加速，大气污染、噪声污染、固体废物处理等问题突出。发展绿色建筑、完善公共交通、推广垃圾分类等举措有助于建设宜居城市。

面对这些挑战，可持续发展理念为解决环境与发展问题提供了思路。可持续发展强调满足当代人需要的同时不损害后代人满足其需要的能力，要求在经济发展、社会进步与环境保护之间寻求平衡。

具体而言，推动可持续发展需要采取以下措施：一是发展绿色低碳循环经济，推动产业结构绿色转型；二是建立健全生态环境保护法律法规和政策体系；三是加强环境科技创新，发展清洁能源、环保材料等绿色技术；四是倡导绿色生活方式，培养公众环保意识；五是加强国际环境合作，共同应对全球环境挑战。

总之，环境保护与可持续发展是一项长期而艰巨的任务，需要政府、企业、社会组织和公众的共同参与和努力，为子孙后代留下蓝天、绿水、青山。""",
            source="环境科学杂志",
            date="2023-06-05"
        ),
        ChineseDocument(
            title="健康生活方式指南",
            content="""在现代快节奏的生活中，维持健康的生活方式对预防疾病、提高生活质量至关重要。健康的生活方式不仅包括合理的饮食和适当的运动，还涉及良好的心理状态和健康的社交关系。

在饮食方面，应遵循平衡膳食原则。《中国居民膳食指南(2022)》建议，每天摄入谷薯类食物200-300克，蔬菜300-500克，水果200-350克，畜禽肉类40-75克，鱼虾类40-75克，蛋类40-50克，奶类300-500克，大豆及坚果类25-35克。减少盐、油、糖的摄入，控制在每天6克、25-30克和25克以下。多喝水，成人每天7-8杯（1500-1700毫升）。避免暴饮暴食，规律进餐，细嚼慢咽。

在运动方面，世界卫生组织建议，成年人每周至少进行150-300分钟中等强度有氧运动，或75-150分钟高强度有氧运动，或等量的组合。此外，每周至少有2天进行全身主要肌肉群的力量训练。对于老年人，还应增加平衡训练和柔韧性练习。选择适合自己的运动方式，如散步、慢跑、游泳、骑车、太极拳等，坚持循序渐进，避免运动损伤。

充足的睡眠是健康的重要组成部分。成年人每晚应保证7-8小时的睡眠时间，青少年需要8-10小时，儿童需要9-12小时。养成规律的作息习惯，避免熬夜，睡前1-2小时不使用电子设备，创造安静、舒适的睡眠环境。

心理健康同样重要。学会压力管理，可以通过冥想、深呼吸、瑜伽等放松技术缓解压力。培养积极的思维方式，学会接纳自己，设定合理的期望。如果出现持续的情绪问题，应及时寻求专业心理咨询或治疗。

社交健康是整体健康的重要部分。保持良好的家庭和社交关系，积极参与社区活动，建立支持网络。研究表明，拥有良好社交关系的人寿命更长，患慢性疾病的风险更低。

此外，戒烟限酒也是健康生活的必要条件。吸烟是多种疾病的主要危险因素，应完全戒除。饮酒应适量，男性每日酒精摄入量不超过25克，女性不超过15克。

定期体检可以早期发现健康问题。根据年龄、性别和健康状况，制定个性化的体检计划，做到疾病早发现、早诊断、早治疗。

最后，保持良好的个人卫生习惯，如勤洗手、定期洗澡、保持口腔卫生等，可以预防多种传染病和慢性疾病。

总之，健康的生活方式需要在饮食、运动、睡眠、心理和社交等多方面平衡发展，通过日常的坚持和努力，可以显著提高生活质量，预防疾病，享受健康长寿。""",
            source="健康教育中心",
            date="2023-07-20"
        ),
    ]


def create_document_corpus(docs: list[ChineseDocument]) -> dict:
    """
    创建文档语料库用于插入GraphRAG
    """
    corpus = {}
    for i, doc in enumerate(docs):
        doc_id = f"doc-{i}"
        corpus[doc_id] = {
            "content": f"标题: {doc.title}\n\n{doc.content}\n\n来源: {doc.source}\n日期: {doc.date}",
            "metadata": {
                "title": doc.title,
                "source": doc.source,
                "date": doc.date
            }
        }
    return corpus


async def insert_documents():
    """异步插入文档到GraphRAG"""
    logger.info("创建GraphRAG实例...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=siliconflow_embedding,
        # 增加块大小以适应中文
        chunk_token_size=1600,
        # 对于中文内容，增加实体提取的最大迭代次数
        entity_extract_max_gleaning=2,
    )
    
    logger.info("准备示例文档...")
    docs = prepare_sample_documents()
    corpus = create_document_corpus(docs)
    
    logger.info(f"开始插入{len(corpus)}篇文档...")
    await rag.ainsert(corpus)
    logger.info("文档插入完成！")


async def query_documents():
    """异步查询GraphRAG"""
    logger.info("加载GraphRAG实例...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=siliconflow_embedding,
    )
    
    # 准备查询
    queries = [
        "人工智能的发展历程是怎样的？",
        "中国传统文化在现代社会有什么价值？",
        "数字经济有哪些主要发展趋势？",
        "环境保护面临哪些主要挑战？",
        "如何保持健康的生活方式？"
    ]
    
    # 使用不同模式进行查询
    for query in queries:
        logger.info(f"\n--- 查询: {query} ---")
        
        # 本地模式
        logger.info("本地模式查询结果:")
        result_local = await rag.aquery(
            query,
            param=QueryParam(
                mode="local",
                response_type="详细的中文回答",
                # 适当增加token限制以获取更完整的回答
                local_max_token_for_text_unit=5000,
                local_max_token_for_community_report=4000,
            )
        )
        print(result_local)
        print("\n" + "-"*80 + "\n")
        
        # 全局模式
        logger.info("全局模式查询结果:")
        result_global = await rag.aquery(
            query,
            param=QueryParam(
                mode="global",
                response_type="详细的中文回答",
                global_max_token_for_community_report=16384,
            )
        )
        print(result_global)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # 创建工作目录
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # 获取事件循环
    loop = asyncio.get_event_loop()
    
    # 如果没有已处理的数据，则插入文档
    if not os.path.exists(os.path.join(WORKING_DIR, "chunk_entity_relation")):
        loop.run_until_complete(insert_documents())
    
    # 进行查询
    loop.run_until_complete(query_documents()) 