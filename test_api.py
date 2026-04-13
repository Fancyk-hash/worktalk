import json
from openai import OpenAI

client = OpenAI(api_key="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")

# Load workplace vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

def get_relevant_vocab(text):
    """Find vocabulary words that appear in the text"""
    relevant = {}
    text_lower = text.lower()
    
    for category, words in vocab.items():
        for english, spanish in words.items():
            if english.lower() in text_lower or spanish.lower() in text_lower:
                relevant[english] = spanish
    
    return relevant

def translate(text, from_lang, to_lang):
    # Get relevant workplace vocabulary
    relevant_vocab = get_relevant_vocab(text)
    
    # Build vocab hint for the AI
    vocab_hint = ""
    if relevant_vocab:
        vocab_hint = "\nUse these exact translations for workplace terms:\n"
        for eng, esp in relevant_vocab.items():
            vocab_hint += f"- {eng} = {esp}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a workplace translator for CUA facilities department.
Translate from {from_lang} to {to_lang}.
Be natural and concise.{vocab_hint}"""},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# Test WITHOUT vocab (plain translation)
print("WITHOUT RAG:")
print(translate("The floor buffer is near the wet floor sign", "English", "Spanish"))

print("\nWITH RAG (same sentence, vocab injected):")
print(translate("The floor buffer is near the wet floor sign", "English", "Spanish"))

print("\nTest 2:")
print(translate("Necesito más bolsas de basura, llama al supervisor", "Spanish", "English"))