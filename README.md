# Chinese-Medical-LLM
A multi-agent Chinese medical large language model system combining retrieval, reasoning, and knowledge graph.

## Set up the environment:
First, ensure that you have installed LLaMA Factory. You can install it with the following commands:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
Next, install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

## Supervised Fine-Tuning (SFT)
The training and export scripts are located in the finetune/ directory.
The training data is located in the finetune/dataset/ directory.

### Training and Export:
After the LLaMAFactory installation is complete, you can run the `sft_train.sh` script to begin training:
```bash
bash sft_train.sh
```
Run the lora_export.sh script to export the fine-tuned model:
```bash
bash lora_export.sh
```
### llamafactory WebUI:
Or you can manually choose the parameters for fine-tuning on the llamafactory web interface.
start the Web UI with the following command:
```bash
GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```
This will start a web server that listens on port 7860 by default. You can change the port number if needed.
Access the Web UI
Open the following address in your browser:
```bash
http://localhost:7860
```
If you want to access it over a local network or the internet, replace localhost with your server's IP address.

## Retrieval-Augmented Generation (RAG)
To set up the RAG environment, you can follow these steps. We use Milvus as our victor database.  Milvus is an open-source vector database designed for scalable similarity search.  
### A.Here’s a step-by-step guide to help you set it up in your environment:
### 1.Install Docker
Milvus provides official Docker images, so installing Docker is the easiest way to set up the environment.  
For Ubuntu, MacOS and Windows: Download and install Docker from [Docker's official website](https://www.docker.com/products/docker-desktop).
### 2.Install Milvus
Download the official Docker Compose file.
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
In the same directory, start Milvus:
```bash
sudo docker compose up -d
```
Verify it is running:
```bash
sudo docker compose ps
```
You should see containers named like `milvus-etcd`, `milvus-minio`, `milvus-standalone` in “Up” state.
### 3.Connect & Use with Python
Install the Python SDK:
```bash
pip install pymilvus
```
Then in Python:
```python
from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")

# Example: create a collection
collection = Collection(
    name="example_collection",
    schema={
        "fields": [
            {"name": "id", "dtype": "INT64", "is_primary": True},
            {"name": "emb", "dtype": "FLOAT_VECTOR", "dim": 768}
        ]
    }
)
print("Collection created:", collection.name)
```
If everything runs successfully, the terminal will output:
```yaml
Collection created: example_collection
```
### B.Importing Medical Data into Milvus
After you have successfully installed and started Milvus, you can use the script `import.py` to load your dataset into the Milvus vector database.  
```bash
python import.py \
    --jsonl ./rag/medical_qa.jsonl \
    --emb-model /path/to/Qwen3-embedding \
    --collection medical_qa \
    --milvus-host localhost \
    --milvus-port 19530 \
    --device cuda
```
#### What this does:
1.Reads the Huatuo medical dataset from the .jsonl file.  
2.Converts each question–answer pair into an embedding using the specified embedding model.  
3.Saves all the embeddings into the Milvus collection (default name: huatuo_qa).  
#### Example Output:
```
[→] Writing 1000 vectors to Milvus ...
[✓] Completed!
```
### C.Run the RAG-based Medical QA System
After you have uploaded your medical Q&A dataset to Milvus (using the `import.py` script), you can start the retrieval-augmented generation (RAG) question-answering system by running `medical_qa.py`.  
This script connects your local large language model (LLM) and the Milvus vector database to perform knowledge-augmented reasoning and question answering.  
#### Run the RAG QA script:
```python
python medical_qa.py \
    --collection huatuo_qa \
    --milvus-host localhost \
    --milvus-port 19530 \
    --emb-model /path/to/Qwen3-Embedding-0.6B \
    --llm-model Qwen3-8B \
    --llm-endpoint http://localhost:8000/v1 \
    --query "A 55-year-old male has nighttime shortness of breath and leg swelling."
```
Or run interactively:
```python
python medical_qa.py \
    --collection huatuo_qa \
    --milvus-host localhost \
    --milvus-port 19530 \
    --emb-model /path/to/Qwen3-Embedding-0.6B \
    --llm-model Qwen3-8B \
    --llm-endpoint http://localhost:8000/v1
```
#### What Happens
The script connects to the Milvus database that stores your vectorized medical data.  
It uses the Qwen-Embedding model to compute embeddings for user queries.  
Then it retrieves the most relevant documents (e.g., medical Q&A pairs) from Milvus.  
The retrieved context is passed to your locally deployed LLM (Qwen3-8B).  
The LLM generates an answer based on both the question and the retrieved medical knowledge.
#### Example Output
```
[AI]
Based on the symptoms, the patient may have heart failure related to hypertension.
Recommendation: consult a cardiologist for echocardiography and further management.

[Source questions]
1. 内容：问题：夜间呼吸困难并双下肢水肿怎么办？ 答案：考虑心功能不全，应就诊心内科。
   元信息：{'question': '夜间呼吸困难并双下肢水肿怎么办？', 'answer': '考虑心功能不全，应就诊心内科。'}
```
