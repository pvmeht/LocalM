# test_hf_cache.py
from transformers import Blip2Processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
print("Cache location:", processor.cache_dir)