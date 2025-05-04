import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import re
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data - Fix for punkt_tab error by downloading punkt
try:
    nltk.download('punkt')
except Exception as e:
    st.error(f"Failed to download NLTK data: {e}")

# Set page configuration
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .score-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .score-description {
        color: #000000;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        min-height: 120px;
    }
    .card-title {
        color: #000000 !important;
        font-weight: bold;
    }
    .metric-value {
        color: #000000 !important;
    }
    .results-section {
        color: #000000 !important;
    }
    /* Target only the metric-related elements */
    .results-section, 
    .card-title,
    .metric-value,
    .score-value,
    .score-description {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentence_transformer():
    """Load the sentence transformer model for semantic similarity."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading sentence transformer model: {e}")
        return None

# Initialize models
sentence_model = load_sentence_transformer()
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts using sentence transformers."""
    if not text1 or not text2:
        return 0.0
    
    try:
        # Encode texts to get embeddings
        embedding1 = sentence_model.encode(text1)
        embedding2 = sentence_model.encode(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(similarity)
    except Exception as e:
        st.error(f"Error calculating semantic similarity: {e}")
        return 0.0

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate texts."""
    if not reference or not candidate:
        return 0.0
    
    try:
        # Use a more robust tokenization approach
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # Calculate BLEU score with smoothing
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    except Exception as e:
        st.error(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores between reference and candidate texts."""
    if not reference or not candidate:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        scores = rouge_scorer_instance.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'rouge_l_precision': scores['rougeL'].precision,
            'rouge_l_recall': scores['rougeL'].recall,
            'rouge_l_f1': scores['rougeL'].fmeasure
        }
    except Exception as e:
        st.error(f"Error calculating ROUGE scores: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}

def calculate_answer_relevancy(question, answer, context):
    """Calculate how relevant the answer is to the question and context."""
    if not question or not answer or not context:
        return 0.0
    
    try:
        # This is a simplified approach - a more sophisticated approach would use 
        # a QA model to determine relevancy
        question_answer_similarity = calculate_semantic_similarity(question, answer)
        context_answer_similarity = calculate_semantic_similarity(context, answer)
        
        # Weighted average of the two similarities
        relevancy = 0.7 * question_answer_similarity + 0.3 * context_answer_similarity
        return relevancy
    except Exception as e:
        st.error(f"Error calculating answer relevancy: {e}")
        return 0.0

def calculate_faithfulness(generated_answer, retrieved_context):
    """Calculate faithfulness - how well the answer relies on the retrieved context."""
    if not generated_answer or not retrieved_context:
        return 0.0
    
    try:
        # For simplicity and to avoid NLTK errors, use simple sentence splitting
        context_sentences = [s.strip() for s in retrieved_context.split('.') if len(s.strip()) > 10]
        
        # Calculate how many of these facts are represented in the answer
        faithfulness_scores = []
        for sentence in context_sentences:
            if len(sentence) > 10:  # Filter out very short sentences
                similarity = calculate_semantic_similarity(sentence, generated_answer)
                faithfulness_scores.append(similarity)
        
        if not faithfulness_scores:
            return 0.0
            
        # Take the average of the top 3 most similar sentences (or all if fewer than 3)
        faithfulness_scores.sort(reverse=True)
        top_scores = faithfulness_scores[:min(3, len(faithfulness_scores))]
        return sum(top_scores) / len(top_scores)
    except Exception as e:
        st.error(f"Error calculating faithfulness: {e}")
        return 0.0

def calculate_contextual_metrics(reference_context, retrieved_context, generated_answer):
    """Calculate contextual precision, recall, and relevancy."""
    if not reference_context or not retrieved_context or not generated_answer:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'relevancy': 0.0
        }
    
    try:
        # For simplicity and to avoid NLTK errors, use simple sentence splitting
        ref_chunks = [s.strip() for s in reference_context.split('.') if len(s.strip()) > 10]
        ret_chunks = [s.strip() for s in retrieved_context.split('.') if len(s.strip()) > 10]
        
        # Filter out very short sentences
        ref_chunks = [chunk for chunk in ref_chunks if len(chunk) > 10]
        ret_chunks = [chunk for chunk in ret_chunks if len(chunk) > 10]
        
        if not ref_chunks or not ret_chunks:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'relevancy': 0.0
            }
        
        # Calculate chunk-wise similarities
        chunk_similarities = []
        for ret_chunk in ret_chunks:
            chunk_best_sim = max([calculate_semantic_similarity(ret_chunk, ref_chunk) for ref_chunk in ref_chunks])
            chunk_similarities.append(chunk_best_sim)
        
        # Precision: What fraction of retrieved chunks are relevant?
        precision = sum(sim > 0.7 for sim in chunk_similarities) / len(ret_chunks)
        
        # Recall: What fraction of reference chunks are covered by retrieved chunks?
        recall_scores = []
        for ref_chunk in ref_chunks:
            recall_best_sim = max([calculate_semantic_similarity(ref_chunk, ret_chunk) for ret_chunk in ret_chunks])
            recall_scores.append(recall_best_sim > 0.7)
        recall = sum(recall_scores) / len(ref_chunks)
        
        # Relevancy: How relevant is the retrieved context to the generated answer?
        relevancy = calculate_semantic_similarity(retrieved_context, generated_answer)
        
        return {
            'precision': precision,
            'recall': recall,
            'relevancy': relevancy
        }
    except Exception as e:
        st.error(f"Error calculating contextual metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'relevancy': 0.0
        }

def create_radar_chart(metrics):
    """Create a radar chart for metrics visualization."""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line_color='rgb(37, 99, 235)',
        fillcolor='rgba(37, 99, 235, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title={
            "text": "Metrics Radar Chart",
            "font": {"color": "black"}
        },
        font={"color": "black"}
    )
    
    return fig

def calculate_overall_score(metrics):
    """Calculate an overall score based on all metrics."""
    # Define weights for different metric groups
    weights = {
        'semantic_similarity': 0.15,
        'bleu_score': 0.05,
        'rouge_l_f1': 0.1,
        'answer_relevancy': 0.2,
        'faithfulness': 0.2,
        'contextual_precision': 0.1,
        'contextual_recall': 0.1,
        'contextual_relevancy': 0.1
    }
    
    overall_score = sum(weights[key] * metrics[key] for key in weights)
    return overall_score

def display_metric_card(title, value, description=""):
    """Display a metric in a visually appealing card."""
    st.markdown(f"""
    <div class="metric-card">
        <h3 class="card-title" style="color: black !important;">{title}</h3>
        <div class="score-value metric-value" style="color: black !important;">{value:.4f}</div>
        <p class="score-description" style="color: black !important;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">LLM Evaluation Dashboard</h1>', unsafe_allow_html=True)
    
    # Main content area - single frame layout
    st.markdown('<h2 class="section-header">Input Data</h2>', unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        reference_response = st.text_area("Reference Response (ground truth answer)", 
                                          height=120,
                                          help="Enter the correct answer (ground truth)")
        
        reference_context = st.text_area("Reference Context (ground truth context or source)", 
                                         height=120,
                                         help="Enter the correct context that should be used")
    
    with col2:
        generated_response = st.text_area("Generated Response (model output)", 
                                          height=120,
                                          help="Enter the response generated by your model")
        
        retrieved_context = st.text_area("Retrieved Context (context used by model)", 
                                         height=120,
                                         help="Enter the context actually retrieved and used by your model")
    
    # Removed the Question (optional) field as requested
    
    evaluate_button = st.button("Evaluate", use_container_width=True)
    
    if evaluate_button:
        if not reference_response or not generated_response:
            st.error("Please provide at least the reference response and generated response.")
        else:
            with st.spinner("Calculating metrics..."):
                # Calculate all metrics
                metrics = {}
                
                # Semantic similarity
                metrics['semantic_similarity'] = calculate_semantic_similarity(reference_response, generated_response)
                
                # BLEU score
                metrics['bleu_score'] = calculate_bleu_score(reference_response, generated_response)
                
                # ROUGE scores
                rouge_scores = calculate_rouge_scores(reference_response, generated_response)
                metrics['rouge_l_precision'] = rouge_scores['rouge_l_precision']
                metrics['rouge_l_recall'] = rouge_scores['rouge_l_recall']
                metrics['rouge_l_f1'] = rouge_scores['rouge_l_f1']
                
                # Context and relevancy metrics (if contexts are provided)
                if reference_context and retrieved_context:
                    # Answer relevancy
                    metrics['answer_relevancy'] = calculate_answer_relevancy(
                        reference_response,  # Using reference_response instead of question
                        generated_response,
                        retrieved_context
                    )
                    
                    # Faithfulness
                    metrics['faithfulness'] = calculate_faithfulness(generated_response, retrieved_context)
                    
                    # Contextual metrics
                    contextual_metrics = calculate_contextual_metrics(reference_context, retrieved_context, generated_response)
                    metrics['contextual_precision'] = contextual_metrics['precision']
                    metrics['contextual_recall'] = contextual_metrics['recall']
                    metrics['contextual_relevancy'] = contextual_metrics['relevancy']
                else:
                    metrics['answer_relevancy'] = 0.0
                    metrics['faithfulness'] = 0.0
                    metrics['contextual_precision'] = 0.0
                    metrics['contextual_recall'] = 0.0
                    metrics['contextual_relevancy'] = 0.0
                
                # Calculate overall score
                metrics['overall_score'] = calculate_overall_score(metrics)
                
                # Display results in a visually appealing dashboard
                st.markdown('<h2 class="section-header results-section">Evaluation Results</h2>', unsafe_allow_html=True)
                
                # Display overall score prominently
                st.markdown(f"""
                <div style="background-color: #E0E7FF; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h2 style="color: black !important;">Overall Score</h2>
                    <div style="font-size: 3rem; font-weight: bold; color: black !important;">{metrics['overall_score']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create columns for metrics display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    display_metric_card("Semantic Similarity", metrics['semantic_similarity'], 
                                        "Measures how similar the generated and reference responses are semantically")
                    display_metric_card("BLEU Score", metrics['bleu_score'], 
                                        "Measures n-gram overlap between responses")
                    display_metric_card("ROUGE-L F1", metrics['rouge_l_f1'], 
                                        "F1 score for longest common subsequence")
                
                with col2:
                    display_metric_card("Answer Relevancy", metrics['answer_relevancy'], 
                                        "How relevant the answer is to the question and context")
                    display_metric_card("Faithfulness", metrics['faithfulness'], 
                                        "How well the answer relies on the retrieved context")
                    display_metric_card("ROUGE-L Precision", metrics['rouge_l_precision'], 
                                        "Precision score for longest common subsequence")
                
                with col3:
                    display_metric_card("Contextual Precision", metrics['contextual_precision'], 
                                        "What fraction of retrieved context is relevant")
                    display_metric_card("Contextual Recall", metrics['contextual_recall'], 
                                        "What fraction of reference context is covered")
                    display_metric_card("ROUGE-L Recall", metrics['rouge_l_recall'], 
                                        "Recall score for longest common subsequence")
                
                # Display visualizations
                st.markdown('<h2 class="section-header results-section">Metrics Visualization</h2>', unsafe_allow_html=True)
                
                # Create two columns for visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Prepare data for radar chart
                    radar_metrics = {
                        'Semantic Similarity': metrics['semantic_similarity'],
                        'BLEU Score': metrics['bleu_score'],
                        'ROUGE-L F1': metrics['rouge_l_f1'],
                        'Answer Relevancy': metrics['answer_relevancy'],
                        'Faithfulness': metrics['faithfulness'],
                        'Contextual Precision': metrics['contextual_precision'],
                        'Contextual Recall': metrics['contextual_recall'],
                        'Contextual Relevancy': metrics['contextual_relevancy']
                    }
                    
                    radar_chart = create_radar_chart(radar_metrics)
                    st.plotly_chart(radar_chart, use_container_width=True)
                
                with viz_col2:
                    # ROUGE Scores Breakdown
                    rouge_fig = go.Figure(data=[
                        go.Bar(
                            name='ROUGE-L Metrics',
                            x=['Precision', 'Recall', 'F1'],
                            y=[metrics['rouge_l_precision'], metrics['rouge_l_recall'], metrics['rouge_l_f1']],
                            marker_color=['#3B82F6', '#10B981', '#8B5CF6']
                        )
                    ])
                    rouge_fig.update_layout(
                        yaxis_range=[0, 1],
                        title={
                            "text": "ROUGE Scores Breakdown",
                            "font": {"color": "black"}
                        },
                        font={"color": "black"}
                    )
                    st.plotly_chart(rouge_fig, use_container_width=True)

if __name__ == "__main__":
    main()