import yaml
import os
from data.api_text_ICEWS18 import QwenAPIGenerator
from dotenv import load_dotenv
load_dotenv()

def test_qwen_api():
    # 1. 读取配置文件
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"错误：找不到配置文件 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_config = config['data']['text_generation']['api']
    
    # 确保 API Key 存在
    api_key = api_config.get('api_key') or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告：未在 config.yaml 或环境变量 DASHSCOPE_API_KEY 中找到 API Key！")
        print("请在终端执行: export DASHSCOPE_API_KEY='你的sk-xxxx'")
        return

    # 2. 初始化生成器
    print("正在初始化 Qwen API 生成器并测试连接...")
    try:
        generator = QwenAPIGenerator(
            api_key=api_key,
            base_url=api_config.get('base_url'),
            cache_dir=config['data']['text_generation']['cache_dir'],
            model=api_config.get('model')
        )
    except Exception as e:
        print(f"初始化失败，请检查网络或 API Key: {e}")
        return

    # 3. 模拟 ICEWS18 的字典映射
    entity_dict = {
        1001: "United States",
        1002: "China",
        1003: "United Nations",
        1004: "European Union"
    }
    
    relation_dict = {
        50: "Express intent to cooperate",
        51: "Make a statement",
        52: "Host a visit",
        53: "Impose embargo, boycott, or sanctions"
    }

    # 4. 模拟几个 ICEWS18 格式的五元组 (s, p, o, time_step, placeholder)
    # 注意 time_step 的测试：
    # 45 -> 2018年第45天 (应该生成 February 14, 2018)
    # 5760 -> 5760小时 -> 240天 -> 2018年第241天 (应该生成 August 29, 2018)
    mock_triples = [
        (1001, 50, 1002, 45, 0),
        (1002, 51, 1003, 5760, 0),
        (1004, 53, 1001, 300, 0)
    ]

    print(f"\n开始为 {len(mock_triples)} 条测试数据生成自然语言文本 (调用模型: {api_config.get('model')})...")

    # 5. 调用批量生成方法
    texts = generator.batch_generate(
        triples=mock_triples,
        entity_dict=entity_dict,
        relation_dict=relation_dict,
        requests_per_minute=api_config.get('requests_per_minute', 60),
        concurrency=api_config.get('concurrency', 2),
        triples_per_request=5, # 测试时设置大一点，让这3条数据一次性发过去
        show_progress=False
    )

    # 6. 打印结果
    print("\n" + "="*50)
    print("🎉 生成结果展示 🎉")
    print("="*50)
    
    for i, text in enumerate(texts):
        s, p, o, time_step, *_ = mock_triples[i]
        print(f"🟢 [测试样例 {i+1}]")
        print(f"输入结构: 主语 [{entity_dict[s]}] | 动作 [{relation_dict[p]}] | 宾语 [{entity_dict[o]}] | 时间步 [{time_step}]")
        print(f"大模型输出: {text}")
        print("-" * 50)

if __name__ == "__main__":
    test_qwen_api()