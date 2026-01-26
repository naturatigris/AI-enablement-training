from nemoguardrails.actions import action
from typing import Optional

@action()
async def check_jailbreak(context: Optional[dict] = None) -> str:
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
    
    user_lower = user_message.lower()
    
    for pattern in jailbreak_patterns:
        if pattern in user_lower:
            print(f"⚠️  Jailbreak pattern detected: {pattern}")
            return "yes"
    
    return "no"


@action()
async def check_moderation(context: Optional[dict] = None) -> str:
    """Check if bot response is appropriate"""
    bot_message = context.get("bot_message", "") if context else ""
    
    # Simple heuristic check for inappropriate content
    blocked_words = [
        "illegal",
        "violence",
        "hate",
        "discrimination",
        "abuse"
    ]
    
    bot_lower = bot_message.lower()
    
    for word in blocked_words:
        if word in bot_lower:
            print(f"⚠️  Inappropriate content detected: {word}")
            return "no"
    
    return "yes"
