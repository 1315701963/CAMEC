#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
huatuo_qa_server_api.py
-----------------------
1. 连接已存在的 Milvus 向量库
    docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.5.12  milvus run standalone
2. 通过 OpenAI 兼容 API（vLLM serve）调用 Qwen3-8B 做检索增强问答
   - 支持单次查询 (--query) 或交互式 REPL
docker run -d --name milvus -p 19530:19530 -p 9091:9091 -v $PWD/milvus_data:/var/lib/milvus milvusdb/milvus:v2.5.2
示例（单次查询）:
    python huatuo_qa21.py \
        --collection huatuo_qa \
        --milvus-host localhost --milvus-port 19530 \
        --emb-model /home/zf1/WuYukang/AIstorian/model/Qwen3-Embedding-0.6B \
        --llm-model /home/zf1/WuYukang/AIstorian/model/lora_export_stage4 \
        --llm-endpoint http://localhost:8000/v1 \
        --query "一位55岁男性夜间呼吸困难并双下肢水肿，伴有既往高血压病史"

示例（交互式）:
    python huatuo_qa_server_api.py [同上参数，不加 --query]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus
from langchain_community.llms.vllm import VLLMOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Milvus 连接 ----------
def connect_vector_store(
    emb_model_path: str,
    device: str,
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
) -> Milvus:
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model_path,
        encode_kwargs={"batch_size": 1, "normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )

    vecstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": milvus_host, "port": milvus_port},
        collection_name=collection_name,
    )
    try:
        test_docs = vecstore.similarity_search("测试", k=1)
        if not test_docs:
            raise RuntimeError(f"集合 '{collection_name}' 中没有向量，请先运行 build_huatuo_index.py")
    except Exception as e:
        raise RuntimeError(f"向量库连接失败：{e}")
    return vecstore


# ---------- LLM & QA Chain ----------
def build_qa_chain(
    vecstore: Milvus,
    llm_endpoint: str,
    llm_model_name: str,
    top_k: int,
) -> RetrievalQA:
    """
    通过 vLLM OpenAI-compatible API 调用推理
    """
    llm = VLLMOpenAI(
        # OpenAI 协议参数
        base_url=llm_endpoint,
        model_name=llm_model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        # 推理超参
        temperature=0.7,
        # max_tokens=1024,
    )

    retriever = vecstore.as_retriever(search_kwargs={"k": top_k})

    prompt_template = """\
    你是一位经验丰富的泌尿科主治医生，请结合提供的医学资料，用简洁且专业的口吻回答患者问题。
    要求：
    • 先给出结论，再给出分析与建议，层次分明。
    • 仅当答案中确有依据时才提出诊疗方案；如需就医，请说明应挂的科室。
    • 回答必须基于以下参考资料，若资料不足请说明“需要进一步检查”。

    【参考资料】
    {context}

    【患者提问】
    {question}

    【医生回答】
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def do_query(chain: RetrievalQA, query: str) -> None:
    result = chain({"query": query})
    print(f"\n[AI]\n{result['result']}\n")
    print("[Source questions]")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"{i}. 内容：{doc.page_content}")
        print(f"   元信息：{doc.metadata}\n")
    #for i, doc in enumerate(result["source_documents"], 1):
    #    print(f"{i}. {doc.metadata['question']}")


# ---------- 命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="huatuo_qa", help="Milvus 集合名")
    parser.add_argument("--milvus-host", default="localhost")
    parser.add_argument("--milvus-port", default="19530")
    parser.add_argument("--emb-model", required=True, help="Qwen-Embedding 模型路径")

    # LLM 相关
    parser.add_argument("--llm-model", required=True, help="推理服务中的 model_name（如 Qwen3-8B）")
    parser.add_argument(
        "--llm-endpoint",
        default="http://localhost:8000/v1",
        help="vLLM serve 的 OpenAI 兼容地址，结尾带 /v1",
    )

    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=3, help="检索条数")
    parser.add_argument("--query", help="若提供则单次查询后退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vecstore = connect_vector_store(
        emb_model_path=args.emb_model,
        device=args.device,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        collection_name=args.collection,
    )

    qa_chain = build_qa_chain(
        vecstore,
        llm_endpoint=args.llm_endpoint,
        llm_model_name=args.llm_model,
        top_k=args.top_k,
    )

    if args.query:
        do_query(qa_chain, args.query)
        return

    # 交互式 REPL
    print("进入交互式问答，输入 'exit' / 'quit' 退出。")
    try:
        while True:
            user_q = input("\n[USER] ").strip()
            if user_q.lower() in {"exit", "quit"}:
                break
            if user_q:
                do_query(qa_chain, user_q)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n再见！")


if __name__ == "__main__":
    main()

