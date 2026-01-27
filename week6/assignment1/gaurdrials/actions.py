from nemoguardrails.actions import action
from typing import Optional

@action()
async def check_jailbreak(context: Optional[dict] = None) -> bool:
    """Check if user message is a jailbreak attempt"""
    user_message = context.get("user_message", "") if context else ""
    
    # Simple heuristic check for jailbreak patterns
    jailbreak_patterns = [
        "ignore your instructions",
        "forget your constraints",
        "pretend you are",
        "you are now",
        "new instructions",
        "bypass",
        "override",
        "jailbreak"
    ]
    harmful_keywords = [
        "how to hack", "teach me to hack", "hacking tutorial",
        "exploit vulnerability", "create malware", 
        "sql injection", "ddos attack",
        "brute force password", "crack password",
        "bypass security", "penetration testing tutorial"
    ]
    
    
    
    user_lower = user_message.lower()
    
    for pattern in jailbreak_patterns:
        if pattern in user_lower:
            print(f"⚠️  Jailbreak pattern detected: {pattern}")
            return True
    for pattern in harmful_keywords:
        if pattern in user_lower:
            print(f"⚠️  Jailbreak pattern detected: {pattern}")
            return True
    
    return False





@action(is_system_action=True)
async def check_if_output_is_safe(context: Optional[dict] = None):
    """Check if bot output contains harmful content"""
    if context is None:
        return True
    
    # Get the last bot message from context
    bot_message = context.get("last_bot_message", "")
    
    # Handle None case
    if bot_message is None:
        bot_message = ""
    
    bot_message_lower = str(bot_message).lower()
    
    harmful_keywords = [
        "hack", "exploit", "malware", "vulnerability",
        "bypass", "injection", "penetration", "brute force",
        "crack", "root access", "privilege escalation"
    ]
    
    for keyword in harmful_keywords:
        if keyword in bot_message_lower:
            return False
    
    return True