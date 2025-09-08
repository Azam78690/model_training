"""
Braille mapping utilities for 6-dot braille patterns
"""

# Standard 6-dot braille mapping
BRAILLE_MAP = {
    (1,0,0,0,0,0): 'A',
    (1,1,0,0,0,0): 'B',
    (1,0,0,1,0,0): 'C',
    (1,0,0,1,1,0): 'D',
    (1,0,0,0,1,0): 'E',
    (1,1,0,1,0,0): 'F',
    (1,1,0,1,1,0): 'G',
    (1,1,0,0,1,0): 'H',
    (0,1,0,1,0,0): 'I',
    (0,1,0,1,1,0): 'J',
    (1,0,1,0,0,0): 'K',
    (1,1,1,0,0,0): 'L',
    (1,0,1,1,0,0): 'M',
    (1,0,1,1,1,0): 'N',
    (1,0,1,0,1,0): 'O',
    (1,1,1,1,0,0): 'P',
    (1,1,1,1,1,0): 'Q',
    (1,1,1,0,1,0): 'R',
    (0,1,1,1,0,0): 'S',
    (0,1,1,1,1,0): 'T',
    (1,0,1,0,0,1): 'U',
    (1,1,1,0,0,1): 'V',
    (0,1,0,1,1,1): 'W',
    (1,0,1,1,0,1): 'X',
    (1,0,1,1,1,1): 'Y',
    (1,0,1,0,1,1): 'Z',
    (0,0,0,0,0,0): ' ',  # Space
}

def braille_to_char(pattern):
    """
    Convert 6-dot braille pattern to character
    
    Args:
        pattern: List or tuple of 6 binary values
        
    Returns:
        Character or '?' if pattern not found
    """
    return BRAILLE_MAP.get(tuple(pattern), '?')

def char_to_braille(char):
    """
    Convert character to 6-dot braille pattern
    
    Args:
        char: Single character
        
    Returns:
        List of 6 binary values or None if character not found
    """
    for pattern, character in BRAILLE_MAP.items():
        if character == char.upper():
            return list(pattern)
    return None

def get_all_patterns():
    """Get all available braille patterns"""
    return BRAILLE_MAP.copy()
