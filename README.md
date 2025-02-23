# PRAGyan_with_CUDA

# CUDA-Optimized RAG with Knowledge Graphs ⚡

**A high-performance Retrieval-Augmented Generation (RAG) system leveraging CUDA-accelerated workflows for knowledge graph operations**

## Key Features

### CUDAManager Class
- **Centralized CUDA Orchestration**  
  Unified management of GPU operations and memory lifecycle
- **Batched Embedding Generation**  
  Mixed-precision processing with `torch.cuda.amp.autocast()`
- **Tensor Core Optimization**  
  Matrix-based similarity computations using FP16/FP32 tensor cores
- **Memory Architecture**  
  Coalesced memory access patterns for optimal DRAM utilization

### GPU-Accelerated Workflows
- **Dynamic BERT Embeddings**  
  Asynchronous pipeline for batched text encoding (256+ samples/batch)
- **Node2Vec GPU Search**  
  Graph walk sampling with CUDA-accelerated random walks
- **Model Graph Compilation**  
  Ahead-of-time kernel fusion with `torch.compile()`
- **Data Transfer Optimization**  
  Pinned memory buffers with async CUDA streams

### Advanced CUDA Techniques
- **Mixed Precision Training**  
  Automatic dtype casting for ops with `autocast(enabled=True)`
- **cuDNN Primitive Optimization**  
  Kernel auto-tuning via `torch.backends.cudnn.benchmark = True`
- **Kernel Fusion**  
  Fused similarity score calculation using custom CUDA kernels
- **Memory-Efficient Expansion**  
  Shared memory indexing for neighbor search operations

### Performance Optimizations
- **Parallel Matrix Ops**  
  Batch matrix multiplication with `torch.bmm()`
- **Zero-Copy Embeddings**  
  Direct memory access with `pin_memory=True` loaders
- **TorchScript JIT**  
  Optimized kernel traces via `@torch.jit.script`
- **Stream-Parallel Execution**  
  Concurrent CUDA streams for overlapped compute

## Installation

```bash
# CUDA 11.8+ required
conda create -n cuda_rag python=3.10
conda activate cuda_rag

# Install CUDA toolkit
conda install -c nvidia cuda-toolkit=11.8

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
