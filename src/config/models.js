export const MODELS = {
  ner: {
    id: 'Xenova/bert-base-NER',
    task: 'token-classification',
    dtype: { webgpu: 'fp16', wasm: 'q8' },
  },
  classification: {
    id: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
    task: 'text-classification',
    dtype: { webgpu: 'fp16', wasm: 'q8' },
  },
  qa: {
    id: 'onnx-community/Qwen2.5-0.5B-Instruct',
    task: 'text-generation',
    dtype: { webgpu: 'q4', wasm: 'q4' },
    generation: { max_new_tokens: 64, temperature: 0.2, top_p: 0.9 },
  },
  summarization: {
    id: 'Xenova/distilbart-cnn-6-6',
    task: 'summarization',
    dtype: { webgpu: 'fp16', wasm: 'q8' },
    generation: { max_new_tokens: 96, min_length: 24 },
  },
};
