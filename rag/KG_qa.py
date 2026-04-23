import re, torch, json
from py2neo import Graph
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置参数
# ============================================================
MODEL_NAME = "/home/zf1/WuYukang/AIstorian/model/lora_export_stage2"
DEVICE = "cuda"

# ============================================================
# 加载主模型
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

# ============================================================
# Neo4j 连接
# ============================================================
graph = Graph("bolt://localhost:7687", auth=("neo4j", "abc12345"), name="neo4j")

# ============================================================
# 词表加载
# ============================================================
def load_vocab(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

DISEASES  = load_vocab("/home/zf1/WuYukang/RAGQnASystem/data/ent_aug/疾病.txt")
SYMPTOMS  = load_vocab("/home/zf1/WuYukang/RAGQnASystem/data/ent_aug/疾病症状.txt")
DRUGS     = load_vocab("/home/zf1/WuYukang/RAGQnASystem/data/ent_aug/药品.txt")
CHECKS    = load_vocab("/home/zf1/WuYukang/RAGQnASystem/data/ent_aug/检查项目.txt")
TREATMENT = load_vocab("/home/zf1/WuYukang/RAGQnASystem/data/ent_aug/治疗方法.txt")

# ============================================================
# 实体抽取
# ============================================================
def extract_entities(text):
    entities = {"disease": [], "symptom": [], "drug": [], "check": [], "treatment": []}
    cleaned_text = re.sub(r"\b[A-D]\s", "", text)

    for i in range(len(cleaned_text)):
        for j in range(i + 1, min(len(cleaned_text), i + 16)):
            word = cleaned_text[i:j + 1]
            if word in DISEASES:
                entities["disease"].append(word)
            elif word in SYMPTOMS:
                entities["symptom"].append(word)
            elif word in DRUGS:
                entities["drug"].append(word)
            elif word in CHECKS:
                entities["check"].append(word)
            elif word in TREATMENT:
                entities["treatment"].append(word)
    return entities

# ============================================================
# Cypher 查询模板
# ============================================================
CYPHER_TEMPLATES = {
    "desc":        "MATCH (d:Disease {name:$name}) RETURN d.desc AS answer",
    "cause":       "MATCH (d:Disease {name:$name}) RETURN d.cause AS answer",
    "prevent":     "MATCH (d:Disease {name:$name}) RETURN d.prevent AS answer",
    "treatment":   "MATCH (d:Disease {name:$name}) RETURN d.cure_way AS answer",
    "symptom":     "MATCH (d:Disease {name:$name})-[:has_symptom]->(s:Symptom) RETURN collect(s.name) AS answer",
    "drug":        "MATCH (d:Disease {name:$name})-[:common_drug|recommand_drug|drugs_of]->(p:Drug) RETURN collect(p.name) AS answer",
    "check":       "MATCH (d:Disease {name:$name})-[:need_check]->(c:Check) RETURN collect(c.name) AS answer",
    "diet_yes":    "MATCH (d:Disease {name:$name})-[:do_eat|recommand_eat]->(f:Food) RETURN collect(f.name) AS answer",
    "diet_no":     "MATCH (d:Disease {name:$name})-[:no_eat]->(f:Food) RETURN collect(f.name) AS answer",
    "department":  "MATCH (d:Disease {name:$name})-[:belongs_to]->(dep:Department) RETURN dep.name AS answer",
    "complication":"MATCH (d:Disease {name:$name})-[:acompany_with]->(c:Disease) RETURN collect(c.name) AS answer",
}

# ============================================================
# 图谱全属性查询
# ============================================================
def fetch_all_attributes(entity):
    results = {}
    for intent, cypher in CYPHER_TEMPLATES.items():
        try:
            value = graph.run(cypher, name=entity).evaluate()
            if value not in (None, [], ""):
                results[intent] = value
        except Exception:
            pass
    if not results:
        return {}
    return {"entity": entity, "all_attributes": results}

# ============================================================
# 图谱信息压缩
# ============================================================
def compress_graph_info(gi_list, max_entities=3, each_list_top=4, desc_len=100, max_chars=800):
    if not gi_list:
        return "无匹配图谱信息"

    lines = []
    for gi in gi_list[:max_entities]:
        if not isinstance(gi, dict):
            continue
        name  = gi.get("entity", "?")
        attrs = gi.get("all_attributes", {})

        frag = [f"【{name}】"]
        for k in ("desc", "cause"):
            v = attrs.get(k)
            if isinstance(v, str) and v.strip():
                frag.append(f"{k}:{v[:desc_len]}")
        for k in ("symptom", "drug", "check", "complication"):
            v = attrs.get(k)
            if isinstance(v, (list, tuple)) and v:
                frag.append(f"{k}:{'、'.join(map(str, v[:each_list_top]))}")

        lines.append(" | ".join(frag))

    s = "\n".join(lines)
    return s[:max_chars] + "……" if len(s) > max_chars else s

# ============================================================
# Neo4j 检索入口
# ============================================================
def neo4j_search(query):
    entities = extract_entities(query)
    all_entities = (
        list(set(entities.get("disease",   []))) +
        list(set(entities.get("symptom",   []))) +
        list(set(entities.get("drug",      []))) +
        list(set(entities.get("check",     []))) +
        list(set(entities.get("treatment", [])))
    )

    graph_info = []
    for entity in all_entities[:5]:
        res = fetch_all_attributes(entity)
        if res:
            graph_info.append(res)
    return graph_info

# ============================================================
# Expert 3 Prompt 构建
# ============================================================
def format_expert3(example, graph_info):
    prompt = (
        "你是疑难杂症门诊专家，请结合题干和知识图谱回答单选题。\n"
        "请先分析关键信息（1-2句），再给出答案。\n"
        "答案格式：答案：X\n\n"
        f"问题：{example['Question']}\n"
    )
    for opt in example["Options"]:
        prompt += f"{opt['key']}. {opt['value']}\n"
    prompt += f"\n知识图谱：\n{graph_info}\n"
    return prompt

# ============================================================
# 答案提取
# ============================================================
def extract_answer_from_expert(text):
    if not text:
        return ""
    m = re.search(r"答案\s*[:：]\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper()
    m = re.search(r"[选是为]\s*[:：]?\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:^|[\s，。、])\s*([A-D])\s*(?:$|[\s，。、])", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"[A-D]", text.upper())
    return m.group(0) if m else ""

# ============================================================
# Expert 3 推理入口
# ============================================================
def run_expert3(example):
    """
    输入: example 字典，包含 "Question" 和 "Options"
    输出: {"answer": "A/B/C/D", "text": 原始输出, "graph_info": 图谱文本}
    """
    query = example["Question"] + " " + " ".join(
        f"{o['key']} {o['value']}" for o in example["Options"]
    )

    # Step 1: 知识图谱检索
    try:
        graph_info_raw  = neo4j_search(query)
        graph_info_text = compress_graph_info(graph_info_raw)
    except Exception as e:
        graph_info_text = f"图谱查询失败: {e}"

    # Step 2: 构建 Prompt 并推理
    prompt = format_expert3(example, graph_info_text)

    with torch.no_grad():
        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **inp,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        text = tokenizer.decode(
            out[0][inp["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        del inp, out
    torch.cuda.empty_cache()

    answer = extract_answer_from_expert(text)
    return {"answer": answer, "text": text, "graph_info": graph_info_text}


# ============================================================
# 单条测试示例
# ============================================================
if __name__ == "__main__":
    example = {
        "Question": "患者男性，58岁，反复咳嗽、咳痰20年，加重伴喘息5年，最可能的诊断是？",
        "Options": [
            {"key": "A", "value": "支气管哮喘"},
            {"key": "B", "value": "慢性阻塞性肺疾病"},
            {"key": "C", "value": "支气管扩张"},
            {"key": "D", "value": "肺结核"},
        ]
    }

    result = run_expert3(example)
    print(f"图谱信息:\n{result['graph_info']}\n")
    print(f"模型输出:\n{result['text']}\n")
    print(f"最终答案: {result['answer']}")
