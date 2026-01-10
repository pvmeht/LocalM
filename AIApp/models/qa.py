# models/qa.py

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logger = logging.getLogger(__name__)

# Load FLAN-T5-Large (768M params — smarter, faithful to context)
try:    
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")
    logger.info("Loaded flan-t5-large for faithful QA")
    IS_T5 = True
except Exception as e:
    logger.warning(f"FLAN-T5 failed ({e}), falling back to gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    IS_T5 = False

def answer_question(question: str, context: str) -> str:
    """
    Generate faithful answer using context only.
    """
    # Anti-hallucination prompt: Strictly instruct to use context only
    prompt = (
        "Answer the question using ONLY the information from the provided context. "
        "Do not add any external knowledge or make up details. "
        "If the context does not have enough information, say 'Not enough information in the resume.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        # Tokenize safely
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=800,  # Safe for both models
            padding=True
        ).to(model.device)

        input_length = inputs["input_ids"].shape[1]
        logger.info(f"Input tokens: {input_length}")

        with torch.no_grad():
            if IS_T5:
                # For T5: Use decoder for generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,  # Low temp = less creative/hallucinatory
                    do_sample=False,  # Deterministic for faithfulness
                    num_beams=4,      # Beam search = better quality
                    early_stopping=True
                )
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                # GPT2 fallback
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_ids = outputs[0][input_length:]
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Post-check: If answer seems off, fallback to direct extraction
        if "passion" in answer.lower() or len(answer) < 20:  # Heuristic for hallucination
            # Direct extract: Pull key lines from context (your "extract same text" idea)
            lines = [line.strip() for line in context.split('\n') if line.strip() and ('Omkar' in line or '@' in line or 'linkedin' in line)]
            extracted = "\n".join(lines[:10])  # Limited to avoid truncation
            return f"Extracted from resume: {extracted[:500]}..."  # Truncate if too long

        return answer

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "Error generating answer. Extracted summary: Omkar Gadhave, Pune... (check logs)"