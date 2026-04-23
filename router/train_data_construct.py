import random
import json
import openai
from datasets import load_dataset
import os

# Initialize the OpenAI API client
client = openai.OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Load the CMExam dataset
START = 0
ds = load_dataset("fzkuji/CMExam", split=f"test[{START}:]")
print(f"Loaded {len(ds)} questions")

# Convert the dataset to a list before sampling
ds_list = ds.to_list()
# print(ds_list[0])  # Print the first sample to check the structure

# Function to annotate the complexity of each question
def gpt4_complexity_annotation(sample):
    prompt = f"""
    作为医学专家，请根据医学诊断复杂度模型（Medical Complexity Score, MCS）对以下问题进行评估。

    问题：{sample['Question']}

    请从以下维度打分（0-3分）：
    1. 症状模糊度Symptom Ambiguity：症状是否非特异性、描述是否完整
    2. 疾病罕见度Disease Rarity：涉及的疾病是否罕见
    3. 并发症风险Comorbidity Risk：是否涉及多系统、有既往病史
    4. 诊断不确定性Diagnostic Uncertainty：鉴别诊断数量、需要的检查复杂度
    [最终复杂度分级规则（基于加权和 MCS）]
    MCS = 0.3·Ambiguity + 0.25·Rarity + 0.25·Comorbidity + 0.2·Uncertainty
        - LOW (0 ≤ MCS < 3): 单一症状，常见病，无并发症，诊断较明确
        - MEDIUM (3 ≤ MCS < 6): 多症状、需要鉴别诊断，一般复杂度
        - HIGH (6 ≤ MCS ≤ 9): 症状复杂、罕见病、多系统问题、诊断依赖检查

    返回严格JSON格式，不要输出额外内容：
    {{
        "symptom_ambiguity": <score>,
        "disease_rarity": <score>,
        "comorbidity_risk": <score>,
        "diagnostic_uncertainty": <score>,
        "overall_level": <"LOW"/"MEDIUM"/"HIGH">,
        "reasoning": "<简短解释>"
        "Question": {sample['Question']}
    }}
    """

    # Get the response from the OpenAI model
    response = client.chat.completions.create(
        model="qwen3-235b-a22b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )

    # Access the content in the response correctly
    message_content = response.choices[0].message.content

    # Print the response content to debug
    print(f"Annotation returned by GPT-4: {message_content}")

    # Clean the message by removing the backticks (```) and extra spaces
    cleaned_content = message_content.strip().lstrip('```json').rstrip('```')

    return cleaned_content


# Annotate 1000 samples randomly
for sample in random.sample(ds_list, 1000):  # Use the list version of ds
    annotation = gpt4_complexity_annotation(sample)

    # Check if the annotation is empty or not
    if annotation.strip():  # Ensure it's not an empty or whitespace-only string
        try:
            # Assume the result is a JSON string, parse it into a dictionary
            annotation_dict = json.loads(annotation)
            annotation_dict['Question'] = sample['Question']  # Add the question to the annotation

            # Save the annotation immediately to the file
            with open('/home/zf1/WuYukang/project/complexity_labels.json', 'a', encoding='utf-8') as f:
                json.dump(annotation_dict, f, ensure_ascii=False, indent=4)
                f.write("\n")  # Write a newline between JSON objects
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error for sample: {annotation}")
    else:
        print(f"Empty annotation for sample {sample['Question']}")

print("Annotations saved to 'complexity_labels.json'")
