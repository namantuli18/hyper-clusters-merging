# Hyperboloid Clusters Merging

This repository contains the code, and was developed as a part of the Assignment 4 for the [CMU ANLP Course 11-711](https://www.phontron.com/class/anlp-fall2024/).


## Components

| Model Name | Merging Notebook | HF Model |
|----------|----------|----------|
| LLaMa2-7B    | [Llama2_7b_hyperboloid_merging](https://github.com/namantuli18/hyper-clusters-merging/blob/main/merge/Llama2_7b_hyperboloid_merging.ipynb)     | [namannn/llama2-7b-hyperbolic-cluster-pruned](https://huggingface.co/namannn/llama2-7b-hyperbolic-cluster-pruned)
| LLaMa2-13B    | [Llama2_13b_hyperboloid_merging](https://github.com/namantuli18/hyper-clusters-merging/blob/main/merge/Llama2_13b_hyperboloid_merging.ipynb)     | [namannn/llama2-13b-hyperbolic-cluster-pruned](https://huggingface.co/namannn/llama2-13b-hyperbolic-cluster-pruned)
| Baichuan2-7B    | [Baichuan2_7b_hyperboloid_merging](https://github.com/namantuli18/hyper-clusters-merging/blob/main/merge/Baichuan2_7b_hyperboloid_merging.ipynb)     | [namannn/baichuan2-7b-hyperbolic-cluster-pruned](https://huggingface.co/namannn/baichuan2-7b-hyperbolic-cluster-pruned)


## Merging Overview 

1. Compute layer embeddings:
    - Process text samples from Wikipedia
    - Extract hidden states for each layer
    - Compute average embeddings

2. Project embeddings to hyperboloid:
    - Use exponential map to project Euclidean vectors onto hyperboloid

3. Perform hyperbolic spectral clustering:
    - Compute pairwise similarities using Lorentzian inner product
    - Apply spectral clustering to group similar layers
4. Merge layers:
    - Identify clusters with high similarity
    - Merge layers within each selected cluster


## Setup
### 1. Install required packages:
```bash
pip install torch transformers datasets huggingface_hub geoopt
```
### 2. Login to Hugging Face:
```python
from huggingface_hub import login
login("YOUR_HF_TOKEN")
```

### 3. Initialize Model and Tokenizer
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

llama_path = MODEL_PATH
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_path,
    trust_remote_code=True,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
```

### 4. Run the provided Python script to project hidden states to a hyperboloid:
```python
# Compute layer embeddings
global_hidden_states = process_all_samples(texts, tokenizer, llama_model, chunk_size)
averaged_hidden_states = compute_layer_averages(global_hidden_states, len(texts))

# Project to hyperboloid
proj_embeddings = project_layers_to_hyperboloid(averaged_hidden_states, curv=1.0)
```


### 5. Perform spectral clustering on the generated Affinity Matrix:
```python
# Perform clustering
clusters, labels = hyperbolic_spectral_clustering(proj_embeddings, n_clusters=8)
clusters_dict, similarities, top_clusters = compute_cluster_similarities(clusters, proj_embeddings)
```

### 6. Merge the layers based on their cluster similarities:
```python
# Merge layers
SIMILARITY_THRESHOLD = 500
merged_model = merge_clusters(
    model=llama_model,
    clusters=clusters,
    cluster_similarities=similarities,
    threshold=SIMILARITY_THRESHOLD,
    merge_layers_fn=merge_layers_in_place
)
```

### 7. Generate text outputs:
The pruned model will have fewer layers than the original LLaMA2-13B. You can generate text using the pruned model:

```python
prompt = "Today is a beautiful day"
pruned_text = generate_text(merged_model, tokenizer, prompt)
print("Pruned Model Output:")
print(pruned_text)
```

### 8. Save the merged model:

```python
merged_model.save_pretrained("/path/to/pruned_model")
tokenizer.save_pretrained("/path/to/pruned_model")
```

### 9. Upload model to HF-Hub:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="your-username/model-name", exist_ok=True)
api.upload_folder(
    folder_path="/path/to/pruned_model",
    repo_id="your-username/model-name",
    repo_type="model"
)
```

## Usage/Inference 
The individual models have been uploaded to HuggingFace and can be used directly

- Use with Transformers library
  
  ```python
      # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("namannn/llama2-13b-hyperbolic-cluster-pruned")
    model = AutoModelForCausalLM.from_pretrained("namannn/llama2-13b-hyperbolic-cluster-pruned")
  ```
  
- Use with VLLM

  ```bash
  # Load and run the model:
    vllm serve "namannn/llama2-13b-hyperbolic-cluster-pruned"

  # Call the server using curl:
    curl -X POST "http://localhost:8000/v1/completions" \
    	-H "Content-Type: application/json" \
    	--data '{
    		"model": "namannn/llama2-13b-hyperbolic-cluster-pruned",
    		"prompt": "Once upon a time,",
    		"max_tokens": 512,
    		"temperature": 0.5
    	}'
  ```
