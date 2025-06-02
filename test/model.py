from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Load model for poem analysis
model = AutoModelForCausalLM.from_pretrained(
    "kienhoang123/Llama3.2_Poem_Analysis",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("kienhoang123/Llama3.2_Poem_Analysis")

def analyze_poem(poem, model, tokenizer, max_new_tokens=100):
    """Analyze a poem using the Llama3.2 model to extract emotional tone, metaphor, setting, etc."""
    input_text = f"""
    ### Instruction:
    Analyze the given poem and extract its emotional tone, metaphor, setting, motion, and generate a prompt based on the poem.

    ### Input Poem:
    {poem}

    ### Poem Analysis:
    """
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.2
    )
    full_analysis = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_analysis

def extract_analysis_elements(full_analysis):
    """Extract the key elements from the full analysis text"""
    # Initialize default values
    elements = {
        "emotional_tone": "",
        "metaphor": "",
        "setting": "",
        "motion": ""
    }
    
    # Extract each element using simple text parsing
    lines = full_analysis.split('\n')
    for line in lines:
        line = line.strip()
        if "Emotional Tone:" in line:
            elements["emotional_tone"] = line.split("Emotional Tone:")[1].strip()
        elif "Metaphor:" in line:
            elements["metaphor"] = line.split("Metaphor:")[1].strip()
        elif "Setting:" in line:
            elements["setting"] = line.split("Setting:")[1].strip()
        elif "Motion:" in line:
            elements["motion"] = line.split("Motion:")[1].strip()
    
    # Create a concise analysis string with just the key elements
    concise_analysis = f"""
    Emotion: {elements['emotional_tone']}
    Metaphor: {elements['metaphor']}
    Setting: {elements['setting']}
    Motion: {elements['motion']}
    """
    
    return concise_analysis

def generate_diffusion_prompt(analysis):
    """Generate a diffusion model prompt in Vietnamese based on poem analysis using Ollama API"""
    context = ""  # Initialize empty context for the first call
    
    prompt = f"""
    Bạn là một nghệ sĩ AI chuyên tạo ra các prompt cho mô hình AI tạo hình ảnh (Midjourney, Stable Diffusion).
    Dựa trên phân tích bài thơ sau đây, hãy tạo ra một prompt chi tiết và ngắn gọn bằng tiếng Việt để mô hình AI có thể tạo ra hình ảnh thể hiện được không khí, cảm xúc và ẩn dụ của bài thơ.
    
    Phân tích bài thơ: {analysis}
    
    Prompt cần bao gồm:
    1. Mô tả cảnh vật/khung cảnh chính
    2. Cảm xúc và không khí
    3. Màu sắc chủ đạo
    4. Phong cách nghệ thuật (như tranh sơn dầu, tranh thủy mặc, nhiếp ảnh, v.v.)
    5. Góc nhìn và ánh sáng
    
    Chỉ trả về prompt cuối cùng, không cần giải thích và không sử dụng từ tiếng Anh.
    """
    
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False,
    }
    
    try:
        response = requests.post(
            url='http://localhost:11434/api/generate', json=payload
        ).json()
        return response['response']
    except Exception as e:
        return f"Lỗi khi tạo prompt: {str(e)}"

if __name__ == "__main__":
    sample_poem = """
    đẩy hoa dun lá khỏi tay trời , <
    nghĩ lại tình duyên luống ngậm ngùi . <
    bắc yến nam hồng , thư mấy bức , <
    đông đào tây liễu , khách đôi nơi . <
    lửa ân , dập mãi sao không tắt , <
    biển ái , khơi hoài vẫn chẳng vơi . <
    đèn nguyệt trong xanh , mây chẳng bợn , <
    xin soi xét đến tấm lòng ai ...
    """
    
    # Step 1: Analyze the poem
    print("=== PHÂN TÍCH BÀI THƠ ===")
    full_analysis = analyze_poem(sample_poem, model, tokenizer)
    print(full_analysis)
    
    # Step 2: Extract key elements
    print("\n=== TRÍCH XUẤT YẾU TỐ CHÍNH ===")
    concise_analysis = extract_analysis_elements(full_analysis)
    print(concise_analysis)
    
    # Step 3: Generate diffusion prompt in Vietnamese
    print("\n=== TẠO PROMPT CHO MÔ HÌNH DIFFUSION ===")
    try:
        diffusion_prompt = generate_diffusion_prompt(concise_analysis)
        print(diffusion_prompt)
    except Exception as e:
        print(f"Không thể tạo prompt: {e}")
        print("Vui lòng đảm bảo Ollama đang chạy với mô hình llama3.2")