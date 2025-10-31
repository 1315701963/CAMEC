#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
from typing import List

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus


def load_huatuo(jsonl_path: str) -> List[Document]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            answer = sample["answers"][0]
            for q in sample["questions"][0]:
                docs.append(
                    Document(
                        # ➤ 检索用文本：问题 + 答案
                        page_content=f"问题：{q}\n答案：{answer}",
                        # ➤ 元信息：可以分字段存，之后 UI 展示会更方便
                        metadata={"question": q, "answer": answer}
                    )
                )
    return docs


def build_vector_store(
    docs: List[Document],
    emb_model_path: str,
    device: str,
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
) -> None:
    """把向量写入 Milvus；若集合已存在且包含向量则直接返回."""
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model_path,
        encode_kwargs={
            "batch_size": 2,
            "normalize_embeddings": True,
        },
        model_kwargs={"device": device},
    )

    try:
        vecstore = Milvus(
            embedding_function=embeddings,
            connection_args={"host": milvus_host, "port": milvus_port},
            collection_name=collection_name,
        )
        if vecstore.collection.num_entities > 0:
            print(f"[✓] Collection '{collection_name}' 已存在，跳过写入。")
            return
    except Exception:
        # 集合不存在则继续创建
        pass

    print(f"[→] 开始写入 {len(docs)} 条向量到 Milvus ...")
    Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"host": milvus_host, "port": milvus_port},
        collection_name=collection_name,
        index_params={
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        },
    )
    print("[✓] 写入完成！")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Huatuo JSONL 数据集路径")
    parser.add_argument("--collection", default="huatuo_qa", help="Milvus 集合名")
    parser.add_argument("--milvus-host", default="localhost")
    parser.add_argument("--milvus-port", default="19530")
    parser.add_argument("--emb-model", required=True, help="Qwen-Embedding 模型路径")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_huatuo(args.jsonl)
    build_vector_store(
        docs,
        emb_model_path=args.emb_model,
        device=args.device,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()

