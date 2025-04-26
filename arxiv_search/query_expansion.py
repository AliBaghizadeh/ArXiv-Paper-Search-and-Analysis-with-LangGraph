"""
Query expansion module for enhancing research queries.

This module uses few-shot prompting with domain-specific examples to expand
research queries with relevant concepts and terminology.
"""

import re
from typing import List, Dict, Any, Optional

def expand_research_query(query: str, model: Any) -> str:
    """
    Expand a research query using few-shot prompting.
    
    Uses domain-specific examples to help the model generate
    a comprehensive expansion of the original query.
    
    Args:
        query: The original research query
        model: Generative AI model instance
        
    Returns:
        An expanded version of the query with additional concepts and terms
    """
    # Vision transformer specific example if the query is about vision transformers
    if "vision transformer" in query.lower() or "vit" in query.lower():
        few_shot_examples = """
        Example 1:
        Query: "vision transformer architecture"
        Expanded: The query is about Vision Transformer (ViT) architectures for computer vision tasks. Key aspects to explore include: original ViT design and patch-based image tokenization; comparison with CNN architectures; attention mechanisms specialized for vision; hierarchical and pyramid vision transformers; efficiency improvements like token pruning and sparse attention; distillation techniques for vision transformers; adaptations for different vision tasks including detection and segmentation; recent innovations addressing quadratic complexity and attention saturation.
        
        Example 2: 
        Query: "how do vision transformers process images"
        Expanded: The query focuses on the internal mechanisms of how Vision Transformers process visual information. Key areas to investigate include: patch embedding processes; position embeddings for spatial awareness; self-attention mechanisms for global context; the role of MLP blocks in feature transformation; how class tokens aggregate information; patch size impact on performance and efficiency; multi-head attention design in vision applications; information flow through vision transformer layers; differences from convolutional approaches to feature extraction.
        """
    else:
        few_shot_examples = """
        Example 1:
        Query: "transformer models for NLP"
        Expanded: The query is about transformer architecture models used in natural language processing. Key aspects to explore include: BERT, GPT, T5, and other transformer variants; attention mechanisms; self-supervision and pre-training approaches; fine-tuning methods; performance on NLP tasks like translation, summarization, and question answering; efficiency improvements like distillation and pruning; recent innovations in transformer architectures.
        
        Example 2:
        Query: "reinforcement learning for robotics"
        Expanded: The query concerns applying reinforcement learning methods to robotic systems. Important areas to investigate include: policy gradient methods; Q-learning variants for continuous control; sim-to-real transfer; imitation learning; model-based RL for robotics; sample efficiency techniques; multi-agent RL for coordinated robots; safety constraints in robotic RL; real-world applications and benchmarks; hierarchical RL for complex tasks.
        
        Example 3:
        Query: "graph neural networks applications"
        Expanded: The query focuses on practical applications of graph neural networks. Key dimensions to explore include: GNN architectures (GCN, GAT, GraphSAGE); applications in chemistry and drug discovery; recommender systems using GNNs; traffic and transportation network modeling; social network analysis; knowledge graph completion; GNNs for computer vision tasks; scalability solutions for large graphs; theoretical foundations of graph representation learning.
        """
    
    prompt = f"""Based on the examples below, expand my research query to identify key concepts, relevant subtopics, and specific areas to explore:

    {few_shot_examples}

    Query: "{query}"
    Expanded:"""
    
    generation_config = {"temperature": 1.0}
    
    response = model.generate_content(prompt, generation_config=generation_config)
    
    return response.text

def extract_domain_specific_terms(query: str, expanded_query: str) -> List[str]:
    """
    Extract domain-specific terms from the query and expanded query.
    
    Args:
        query: Original query
        expanded_query: Expanded query
        
    Returns:
        List of domain-specific terms
    """
    domain_specific_terms = []
    
    # Domain detection logic
    if "vision transformer" in query.lower() or "vit" in query.lower():
        domain_specific_terms = ["Vision Transformer", "ViT", "image patches", 
                          "self-attention", "transformer encoder", 
                          "multi-head attention", "computer vision"]
    elif "graph" in query.lower() and "neural" in query.lower():
        domain_specific_terms = ["Graph Neural Network", "GNN", "node embedding",
                          "message passing", "graph attention", "GraphSAGE"]
    # Add more domain detection as needed
    elif "reinforcement learning" in query.lower() or " rl " in f" {query.lower()} ":
        domain_specific_terms = ["Reinforcement Learning", "RL", "policy gradient", 
                          "Q-learning", "reward function", "MDP", "Markov Decision Process", 
                          "DDPG", "PPO", "TD learning", "actor-critic"]
    elif "large language model" in query.lower() or "llm" in query.lower():
        domain_specific_terms = ["Large Language Model", "LLM", "transformer", 
                          "attention mechanism", "GPT", "BERT", "prompt engineering", 
                          "fine-tuning", "few-shot learning", "instruction tuning"]
    elif "diffusion" in query.lower() and ("model" in query.lower() or "image" in query.lower()):
        domain_specific_terms = ["Diffusion Model", "DDPM", "latent diffusion", 
                          "score-based generative model", "noise prediction", 
                          "reverse diffusion", "U-Net", "text-to-image"]
    elif "robotics" in query.lower() or "robot" in query.lower():
        domain_specific_terms = ["Robotics", "robot learning", "manipulation", 
                          "grasping", "trajectory optimization", "inverse kinematics", 
                          "motion planning", "control policy", "sim2real"]
    elif "recommendation" in query.lower() or "recommender" in query.lower():
        domain_specific_terms = ["Recommender System", "collaborative filtering", 
                          "content-based filtering", "matrix factorization", 
                          "user embedding", "item embedding", "CTR prediction"]
    elif "computer vision" in query.lower() or "image" in query.lower():
        domain_specific_terms = ["Computer Vision", "CNN", "object detection", 
                          "segmentation", "image recognition", "feature extraction", 
                          "SIFT", "ResNet", "Faster R-CNN", "YOLO"]
    elif ("natural language" in query.lower() or "nlp" in query.lower()) and "transformer" not in query.lower():
        domain_specific_terms = ["Natural Language Processing", "NLP", "named entity recognition", 
                          "sentiment analysis", "text classification", "word embedding", 
                          "language model", "sequence-to-sequence", "LSTM", "RNN"]
    elif "generative" in query.lower() or "gan" in query.lower():
        domain_specific_terms = ["Generative Adversarial Network", "GAN", "StyleGAN", 
                          "generator", "discriminator", "adversarial training", 
                          "latent space", "mode collapse", "image synthesis"]
    elif "attention" in query.lower() or "transformer" in query.lower():
        domain_specific_terms = ["Transformer", "attention mechanism", "self-attention", 
                          "multi-head attention", "encoder-decoder", "positional encoding", 
                          "cross-attention", "attention weights"]
    elif "quantum" in query.lower() and ("computing" in query.lower() or "machine learning" in query.lower()):
        domain_specific_terms = ["Quantum Computing", "quantum machine learning", 
                          "quantum circuit", "qubit", "quantum gate", "variational quantum circuit", 
                          "QAOA", "quantum advantage", "quantum supremacy"]
    
    # Generic ML terms for any ML-related query
    if any(term in query.lower() for term in ["machine learning", "neural network", "deep learning", "ai"]):
        generic_ml_terms = ["neural network", "deep learning", "backpropagation", 
                     "gradient descent", "loss function", "activation function", 
                     "hyperparameter tuning", "regularization", "overfitting"]
        domain_specific_terms.extend(generic_ml_terms)
    
    # If no specific domain is detected, extract key terms from the expanded query
    if not domain_specific_terms and expanded_query:
        # Extract potential terms from expanded query
        expanded_lines = expanded_query.split('. ')
        for line in expanded_lines:
            # Find capitalized terms or terms in quotes that might be important concepts
            potential_terms = re.findall(r'([A-Z][a-zA-Z0-9]+([ \-][A-Z][a-zA-Z0-9]+)*)', line)
            quoted_terms = re.findall(r'"([^"]+)"', line)
            
            # Add these as domain terms
            for term in potential_terms:
                if isinstance(term, tuple):
                    term = term[0]  # Extract the actual term from regex match tuple
                if len(term) > 3 and term not in domain_specific_terms:  # Only terms longer than 3 chars
                    domain_specific_terms.append(term)
                    
            domain_specific_terms.extend(quoted_terms)
    
    return domain_specific_terms

def create_search_query(original_query: str, domain_terms: List[str]) -> str:
    """
    Create a search query by combining the original query with domain terms.
    
    Args:
        original_query: Original query
        domain_terms: List of domain-specific terms
        
    Returns:
        Enhanced search query
    """
    if not domain_terms:
        return original_query
        
    expanded_terms = " OR ".join(f'"{term}"' for term in domain_terms)
    search_query = f'"{original_query}" OR ({expanded_terms})'
    
    return search_query 