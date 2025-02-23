# Import necessary libraries
import os
import torch
import re
import string
import pandas as pd
import networkx as nx
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline, BertTokenizer, BertModel
from node2vec import Node2Vec
from textblob import Word
import nltk
import openai
import spacy
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ----------------------
# CUDA Optimization Module
# ----------------------
class CUDAManager:
    """
    Manages CUDA operations and optimizations for tensor computations
    """
    
    def __init__(self):
        self.device = self._get_device()
        self.bert_model = None
        self.bert_tokenizer = None
        self.node2vec_embeddings = None
        self.bert_embeddings = None
        self.nodes_list = None
        self.node_to_index = {}
        
    def _get_device(self):
        """Initialize CUDA device with memory optimization"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking
            torch.cuda.empty_cache()  # Clear unused memory
            return torch.device('cuda')
        raise RuntimeError("CUDA device not available - required for this implementation")

    def initialize_models(self):
        """Load and optimize models for CUDA with mixed precision"""
        with torch.cuda.amp.autocast():  # Enable mixed precision
            # Initialize BERT with optimized settings
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            self.bert_model = torch.compile(self.bert_model)  # Enable graph compilation
            
    def precompute_embeddings(self, kg, node2vec_model):
        """
        Precompute all embeddings on CUDA device using batched processing
        """
        # Node2Vec embeddings
        nodes = list(kg.nodes())
        self.nodes_list = nodes
        self.node_to_index = {node: idx for idx, node in enumerate(nodes)}
        
        # Convert Node2Vec embeddings to CUDA tensor
        node2vec_embeddings = [node2vec_model.wv[node] for node in nodes]
        self.node2vec_embeddings = torch.tensor(node2vec_embeddings, 
                                              dtype=torch.float32,
                                              device=self.device)
        
        # Precompute BERT embeddings in batches
        self.bert_embeddings = self._batch_bert_embeddings(nodes)
        
    def _batch_bert_embeddings(self, texts, batch_size=256):
        """
        Generate BERT embeddings in optimized CUDA batches
        """
        embeddings = []
        self.bert_model.eval()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.bert_tokenizer(
                    batch, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                ).to(self.device)
                
                outputs = self.bert_model(**inputs)
                batch_embeds = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeds)
                
        return torch.cat(embeddings, dim=0)
    
    def gpu_similarity_search(self, query, top_n=10, threshold=0.35):
        """
        CUDA-optimized similarity search with parallel computation
        """
        # Generate query embedding
        query_embed = self._batch_bert_embeddings([query]).squeeze()
        
        # Normalize embeddings for efficient matrix multiplication
        query_norm = F.normalize(query_embed, p=2, dim=0)
        corpus_norm = F.normalize(self.bert_embeddings, p=2, dim=1)
        
        # Compute cosine similarity using matrix multiplication
        similarity_scores = torch.mm(query_norm.unsqueeze(0), corpus_norm.T).squeeze()
        
        # Filter and sort results on GPU
        mask = similarity_scores > threshold
        filtered_scores = similarity_scores[mask]
        sorted_scores, indices = torch.sort(filtered_scores, descending=True)
        
        # Return top results with neighbors
        top_indices = indices[:top_n]
        similar_nodes = [self.nodes_list[i] for i in top_indices.cpu().numpy()]
        return self._expand_with_neighbors(similar_nodes)
    
    def _expand_with_neighbors(self, nodes, threshold=0.4, top_n=5):
        """
        CUDA-accelerated neighbor expansion using Node2Vec embeddings
        """
        # Get indices of input nodes
        node_indices = [self.node_to_index[n] for n in nodes]
        node_embeddings = self.node2vec_embeddings[node_indices]
        
        # Compute all pairwise similarities
        similarity_matrix = F.cosine_similarity(
            node_embeddings.unsqueeze(1),
            self.node2vec_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Find top similar nodes not in original set
        expanded = set(nodes)
        for i, node in enumerate(nodes):
            similarities = similarity_matrix[i]
            mask = (similarities > threshold) & ~torch.isin(
                torch.arange(len(self.nodes_list), 
                torch.tensor(node_indices, device=self.device)
            )
            
            valid_indices = torch.nonzero(mask).squeeze()
            if valid_indices.numel() == 0:
                continue
                
            top_similar = similarities[valid_indices].topk(top_n)
            for idx in valid_indices[top_similar.indices]:
                expanded.add(self.nodes_list[idx.item()])
                
        return list(expanded)

# ----------------------
# Optimized TextPreprocessor (CUDA-aware)
# ----------------------
class TextPreprocessor:
    """(Existing implementation remains the same)"""
    # [Previous implementation unchanged]

# ----------------------
# CUDA-accelerated RelationExtractorKG
# ----------------------
class RelationExtractorKG:
    """
    Optimized relation extraction with CUDA memory management
    """
    
    def __init__(self, peft_model_id):
        self.cuda_manager = CUDAManager()
        self.peft_model_id = peft_model_id
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.kg = nx.Graph()

    def initialize_model(self):
        """Load model with CUDA optimization"""
        # Use bfloat16 for better numerical stability on CUDA
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.peft_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            offload_buffers=True
        ).to(self.cuda_manager.device)
        
        # Enable CUDA graph capture for faster inference
        self.model = torch.compile(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.peft_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipe = pipeline("text-generation", 
                            model=self.model, 
                            tokenizer=self.tokenizer,
                            device=self.cuda_manager.device)

    def extract_relations(self, sentences):
        """(Existing implementation remains the same)"""
        # [Previous implementation unchanged]

    def build_knowledge_graph(self, total_responses):
        """(Existing implementation remains the same)"""
        # [Previous implementation unchanged]

# ----------------------
# CUDA-optimized RAGVectorizer
# ----------------------
class RAGVectorizer:
    """
    Implements CUDA-accelerated retrieval and generation
    """
    
    def __init__(self, kg, openai_api_key):
        self.cuda_manager = CUDAManager()
        self.kg = kg
        self.openai_api_key = openai_api_key
        self.node2vec_model = None
        self.cuda_manager.initialize_models()

    def generate_node2vec_embeddings(self):
        """Generate and optimize graph embeddings on CUDA"""
        node2vec = Node2Vec(self.kg, 
                           dimensions=256,  # Increased for better representation
                           walk_length=50,
                           num_walks=300,
                           workers=4)
        
        self.node2vec_model = node2vec.fit(window=15, 
                                         min_count=1,
                                         batch_words=8)
        
        # Convert embeddings to CUDA tensor
        self.cuda_manager.precompute_embeddings(self.kg, self.node2vec_model)

    def generate_answers_with_kg_rag(self, queries, preprocessed_sentences):
        """CUDA-accelerated RAG pipeline"""
        answers = []
        for query in queries:
            # GPU-accelerated similarity search
            similar_nodes = self.cuda_manager.gpu_similarity_search(query)
            
            # Context processing
            context = self._get_context(similar_nodes, preprocessed_sentences)
            
            # Generate answer with GPU-optimized model
            answers.append(self._generate_answer(query, context))
            
        return answers

    def _get_context(self, nodes, sentences):
        """(Existing mapping implementation remains the same)"""
        # [Previous implementation unchanged]

    def _generate_answer(self, query, context):
        """(Existing OpenAI implementation remains the same)"""
        # [Previous implementation unchanged]

# ----------------------
# Main Function with CUDA Optimization
# ----------------------
def main():
    # Initialize with CUDA awareness
    preprocessor = TextPreprocessor()
    
    # Data loading
    df = pd.read_csv("Your_file.csv")
    sentences = [preprocessor.preprocess_text(t) for t in df['self_text']]
    
    # CUDA-accelerated relation extraction
    relation_extractor = RelationExtractorKG("solanaO/llama3-8b-sft-qlora-re")
    relation_extractor.initialize_model()
    responses = relation_extractor.extract_relations(sentences)
    relation_extractor.build_knowledge_graph(responses)
    
    # CUDA-optimized RAG pipeline
    rag_vectorizer = RAGVectorizer(relation_extractor.kg, "your_openai_key")
    rag_vectorizer.generate_node2vec_embeddings()
    
    # Process queries with CUDA acceleration
    queries = ["Your query list here"]
    answers = rag_vectorizer.generate_answers_with_kg_rag(queries, sentences)
    
    print(f"CUDA-accelerated Answers: {answers}")

if __name__ == "__main__":
    main()
