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
To set up the RAG environment, you can follow these steps. We use Milvus as our victor database. Milvus is an open-source vector database designed for scalable similarity search. Here’s a step-by-step guide to help you set it up in your environment:
Connect & Use with Python 1.Install Docker
Milvus provides official Docker images, so installing Docker is the easiest way to set up the environment.
Connect & Use with PythonInstall Milvus
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
