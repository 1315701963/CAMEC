# Chinese-Medical-LLM
A multi-agent Chinese medical large language model system combining retrieval, reasoning, and knowledge graph.

## Set up the environment:
First, ensure that you have installed LLaMA Factory. You can install it with the following commands:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
Install the required dependencies:
pip install -r requirements.txt

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



