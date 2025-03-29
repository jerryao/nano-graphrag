# SiliconFlow 集成示例

这些示例演示了如何使用 SiliconFlow API 与 nano-graphrag 集成，包括嵌入模型和 LLM 模型。

## 示例文件说明

1. **基础嵌入测试**
   - `minimal_siliconflow_embedding.py`: 测试 SiliconFlow 的嵌入模型 (BAAI/bge-m3)
   - `using_siliconflow_embedding.py`: 在 GraphRAG 中使用 SiliconFlow 的嵌入模型

2. **中文嵌入优化**
   - `using_siliconflow_bge_zh.py`: 使用专为中文优化的 BGE 嵌入模型

3. **LLM 测试**
   - `minimal_siliconflow_llm.py`: 测试 SiliconFlow 的 DeepSeek-R1 LLM 模型

4. **完整集成**
   - `siliconflow_complete_example.py`: 完整的 GraphRAG 实现，同时使用 SiliconFlow 的嵌入和 LLM 模型
   - `chinese_documents_with_siliconflow.py`: 针对中文文档的高级 GraphRAG 示例

5. **简化运行器**
   - `run_siliconflow_graphrag.py`: 简化的运行脚本，测试 API 连接和基本功能

## 使用方法

### 安装依赖

```bash
pip install nano-graphrag
pip install aiohttp numpy
```

### 运行示例

1. **测试嵌入功能**:
```bash
python examples/minimal_siliconflow_embedding.py
```

2. **测试 LLM 功能**:
```bash
python examples/minimal_siliconflow_llm.py
```

3. **运行完整的 GraphRAG**:
```bash
python examples/siliconflow_complete_example.py
```

## 示例配置

所有示例都使用 SiliconFlow API，您需要设置有效的 API 密钥:

```python
SILICONFLOW_API_KEY = "YOUR_API_KEY"
```

## 模型选择

- **嵌入模型**:
  - `BAAI/bge-m3`: 通用嵌入模型，支持多语言，1024维，最大支持8192 tokens
  - `BAAI/bge-large-zh-v1.5`: 专为中文优化的嵌入模型，最大支持512 tokens

- **LLM 模型**:
  - `deepseek-ai/DeepSeek-R1`: 强大的通用 LLM，适合多种任务

## 自定义集成

如果您想在自己的项目中集成 SiliconFlow，可以参考 `siliconflow_complete_example.py` 中的实现方式。关键步骤包括:

1. 定义嵌入函数和 LLM 函数
2. 创建 GraphRAG 实例时指定这些函数
3. 使用 GraphRAG 的标准 API 插入文档和查询

这些示例展示了如何完全不依赖 OpenAI API，使用国内的 SiliconFlow 服务构建强大的 GraphRAG 应用。 