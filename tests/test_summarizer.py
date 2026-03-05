"""
Unit tests for summarizer functionality.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.summarizer import MultilingualSummarizer

@pytest.fixture
def summarizer():
    """Fixture to create summarizer instance."""
    return MultilingualSummarizer()

def test_model_initialization(summarizer):
    """Test that model initializes correctly."""
    assert summarizer.device in ['cuda', 'cpu']
    assert summarizer.model is not None
    assert summarizer.tokenizer is not None

def test_english_summarization(summarizer):
    """Test English text summarization."""
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic 
    cognitive functions that humans associate with the human mind, such as 
    learning and problem solving.
    """
    
    summary = summarizer.summarize(text, 'en')
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(text)  # Summary should be shorter

def test_arabic_summarization(summarizer):
    """Test Arabic text summarization."""
    text = """
    الذكاء الاصطناعي هو سلوك وخصائص معينة تتسم بها البرامج الحاسوبية 
    تجعلها تحاكي القدرات الذهنية البشرية وأنماط عملها. من أهم هذه الخاصيات 
    القدرة على التعلم والاستنتاج ورد الفعل على أوضاع لم تبرمج في الآلة. 
    إلا أن هذا المصطلح جدلي نظرا لعدم توفر تعريف محدد للذكاء.
    """
    
    summary = summarizer.summarize(text, 'ar')
    
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_empty_input_handling(summarizer):
    """Test handling of empty input."""
    with pytest.raises(Exception):
        summarizer.summarize("", 'en')

def test_batch_processing(summarizer):
    """Test batch summarization."""
    texts = [
        "First test text for summarization.",
        "Second test text for summarization."
    ]
    
    summaries = summarizer.batch_summarize(texts, 'en')
    
    assert len(summaries) == 2
    assert all(isinstance(s, str) for s in summaries)