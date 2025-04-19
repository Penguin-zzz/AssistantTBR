import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_chatbot():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model = model, tokenizer=tokenizer)

chatbot = load_chatbot()

def load_books(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []
    
st.set_page_config(page_title = "📚Book Recommendations")
st.title("📚Book Recommendations")
st.markdown("Get **completed book series** suggestions based on your favourite genre and theme!")

genre = st.text_input("Enter a genre: ")
theme = st.text_input("Enter a theme you want to explore: ")

if st.button("Recommend!"):
    books = load_books("owned_books.txt")
    books_list = "\n".join(f"- {book}" for book in books) if books else "Nothing Listed"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that recommends completed book series."},
        {"role": "system", "content": f"I own the following books:\n{books_list}\n\n"
                                      f"Based on the genre '{genre} and theme '{theme}',"
                                      f"recommend a completed book series I might enjoy."}
    ]

    prompt = chatbot.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
    response = chatbot(prompt, max_new_tokens=300, do_sample=True, temperature=0.75)
    recommendation = response[0]['generated_text'].replace(prompt,"").strip()

    st.markdown("### 🎯Recommendations")
    st.success(recommendation)
