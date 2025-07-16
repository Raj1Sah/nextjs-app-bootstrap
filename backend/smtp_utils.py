import smtplib
import logging
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_confirmation_email(recipient: str, full_name: str, interview_date: str, interview_time: str) -> bool:
    """
    Send interview booking confirmation email
    
    Args:
        recipient: Recipient email address
        full_name: Full name of the person
        interview_date: Date of the interview
        interview_time: Time of the interview
        
    Returns:
        Boolean indicating success/failure
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg["Subject"] = "Interview Booking Confirmation"
        msg["From"] = SMTP_USER
        msg["To"] = recipient
        
        # Create HTML and text versions of the email
        text_content = f"""
Hello {full_name},

Thank you for booking an interview with us!

Interview Details:
- Date: {interview_date}
- Time: {interview_time}
- Candidate: {full_name}

Please make sure to be available at the scheduled time. If you need to reschedule, please contact us as soon as possible.

We look forward to speaking with you!

Best regards,
Interview Team
        """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .details {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Interview Booking Confirmation</h2>
        </div>
        
        <p>Hello <strong>{full_name}</strong>,</p>
        
        <p>Thank you for booking an interview with us!</p>
        
        <div class="details">
            <h3>Interview Details:</h3>
            <ul>
                <li><strong>Date:</strong> {interview_date}</li>
                <li><strong>Time:</strong> {interview_time}</li>
                <li><strong>Candidate:</strong> {full_name}</li>
            </ul>
        </div>
        
        <p>Please make sure to be available at the scheduled time. If you need to reschedule, please contact us as soon as possible.</p>
        
        <p>We look forward to speaking with you!</p>
        
        <div class="footer">
            <p>Best regards,<br>
            <strong>Interview Team</strong></p>
        </div>
    </div>
</body>
</html>
        """
        
        # Create MIMEText objects
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        # Attach parts
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        logger.info(f"Confirmation email sent successfully to {recipient}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication failed: {str(e)}")
        return False
    except smtplib.SMTPRecipientsRefused as e:
        logger.error(f"Recipient refused: {str(e)}")
        return False
    except smtplib.SMTPServerDisconnected as e:
        logger.error(f"SMTP server disconnected: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Failed to send confirmation email: {str(e)}")
        return False

def send_notification_email(recipient: str, subject: str, message: str, is_html: bool = False) -> bool:
    """
    Send a general notification email
    
    Args:
        recipient: Recipient email address
        subject: Email subject
        message: Email message content
        is_html: Whether the message is HTML formatted
        
    Returns:
        Boolean indicating success/failure
    """
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = recipient
        
        if is_html:
            msg.set_content(message, subtype='html')
        else:
            msg.set_content(message)
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        logger.info(f"Notification email sent successfully to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send notification email: {str(e)}")
        return False

def test_smtp_connection() -> bool:
    """
    Test SMTP connection and authentication
    
    Returns:
        Boolean indicating if connection is successful
    """
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
        
        logger.info("SMTP connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"SMTP connection test failed: {str(e)}")
        return False

def send_file_processing_notification(recipient: str, file_name: str, status: str, details: Optional[str] = None) -> bool:
    """
    Send notification about file processing status
    
    Args:
        recipient: Recipient email address
        file_name: Name of the processed file
        status: Processing status (success, failed, etc.)
        details: Additional details about the processing
        
    Returns:
        Boolean indicating success/failure
    """
    subject = f"File Processing {status.title()}: {file_name}"
    
    if status.lower() == "success":
        message = f"""
File Processing Completed Successfully

File: {file_name}
Status: {status}

{details if details else 'Your file has been processed and is ready for querying.'}

You can now use the query API to search through your document.
        """
    else:
        message = f"""
File Processing Failed

File: {file_name}
Status: {status}

{details if details else 'There was an error processing your file. Please try again or contact support.'}
        """
    
    return send_notification_email(recipient, subject, message)
