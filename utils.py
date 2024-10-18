"""
This module provides utility functions for creating PDF documents from conversation transcripts.

It contains two main functions:
1. create_conversation_pdf: Creates a PDF from a list of conversation entries.
2. create_conversation_pdf_from_messages: Creates a PDF from a list of conversation messages.

Both functions use the ReportLab library to generate professionally formatted PDF documents
with customized styles for titles, speakers, and content.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from typing import List, Dict, Any

def create_conversation_pdf(convo_transcript_list: List[Dict[str, str]], title: str, output_filename: str) -> None:
    """
    Create a PDF document containing a formatted conversation transcript.

    This function takes a list of conversation entries, a title, and an output filename,
    and generates a PDF document with formatted text. The conversation is presented
    with distinct styles for the title, speakers, and content.

    Args:
        convo_transcript_list (List[Dict[str, str]]): A list of dictionaries, where each dictionary
            represents a single utterance with the speaker as the key and the message as the value.
        title (str): The title of the conversation to be displayed at the top of the PDF.
        output_filename (str): The filename (including path) where the PDF will be saved.

    Returns:
        None: The function saves the PDF file but does not return any value.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=12
    )
    
    speaker_style = ParagraphStyle(
        'Speaker',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.blue,
        fontWeight='bold'
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20
    )
    
    # Build the PDF content
    content: List[Any] = []
    content.append(Paragraph(title, title_style))
    content.append(Spacer(1, 12))
    
    for entry in convo_transcript_list:
        for speaker, message in entry.items():
            content.append(Paragraph(f"{speaker}:", speaker_style))
            content.append(Paragraph(message, content_style))
            content.append(Spacer(1, 6))
    
    # Generate the PDF
    doc.build(content)


def create_conversation_pdf_from_messages(conversation_messages: List[Dict[str, str]], title: str, output_filename: str) -> None:
    """
    Creates a PDF file from a list of conversation messages.

    This function takes a list of conversation messages, a title, and an output filename,
    and generates a PDF document with formatted text. The conversation is presented
    with distinct styles for the title, speakers, and content.

    Args:
        conversation_messages (List[Dict[str, str]]): A list of dictionaries, where each dictionary
            represents a single message with 'role' and 'content' keys.
        title (str): The title of the conversation to be displayed at the top of the PDF.
        output_filename (str): The filename (including path) where the PDF will be saved.

    Returns:
        None: The function saves the PDF file but does not return any value.
    """
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=12
    )
    
    speaker_style = ParagraphStyle(
        'Speaker',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.blue,
        fontWeight='bold'
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20
    )
    
    # Build the PDF content
    content: List[Any] = []
    content.append(Paragraph(title, title_style))
    content.append(Spacer(1, 12))
    
    for message in conversation_messages:
        speaker = message['role'].capitalize()
        message_content = message['content']
        content.append(Paragraph(f"{speaker}:", speaker_style))
        content.append(Paragraph(message_content, content_style))
        content.append(Spacer(1, 6))
    
    # Generate the PDF
    doc.build(content)
