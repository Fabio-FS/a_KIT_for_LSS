import numpy as np


def calculate_response_entropy(response):
    words = response.lower().split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    total_words = len(words)
    entropy = 0
    for count in word_counts.values():
        probability = count / total_words
        entropy -= probability * np.log2(probability)
    
    return entropy

def calculate_lexical_diversity(response):
    words = response.lower().split()
    unique_words = set(words)
    return len(unique_words) / len(words)

def calculate_argument_coherence(response, previous_messages):
    shared_terms = set()
    response_words = set(response.lower().split())
    
    for message in previous_messages:
        message_words = set(message.content.lower().split())
        shared_terms.update(response_words & message_words)
    
    return len(shared_terms) / len(response_words)

def calculate_stance_strength(response):
    strong_words = ['strongly', 'definitely', 'absolutely', 'must', 'need', 'essential']
    weak_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'reconsider']
    
    response_lower = response.lower()
    strong_count = sum(1 for word in strong_words if word in response_lower)
    weak_count = sum(1 for word in weak_words if word in response_lower)
    
    return strong_count - weak_count

def calculate_response_length_ratio(response, previous_messages):
    avg_previous_length = np.mean([len(msg.content) for msg in previous_messages])
    return len(response) / avg_previous_length

def analyze_response(response, previous_messages):
    metrics = {
        'entropy': calculate_response_entropy(response),
        'lexical_diversity': calculate_lexical_diversity(response),
        'coherence': calculate_argument_coherence(response, previous_messages),
        'stance_strength': calculate_stance_strength(response),
        'length_ratio': calculate_response_length_ratio(response, previous_messages)
    }
    return metrics


def calculate_response_specificity(response, previous_messages):
    # Generic phrases that indicate the model is giving up
    generic_phrases = [
        'i think', 'we should', 'it seems', 'perhaps', 'in general',
        'overall', 'basically', 'essentially', 'clearly', 'obviously',
        'let\'s', 'we need to', 'important to', 'focus on'
    ]
    
    response_lower = response.lower()
    generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
    
    # Specific elements: numbers, proper nouns, quoted facts
    has_number = any(char.isdigit() for char in response)
    has_statistic = '%' in response or 'percent' in response.lower()
    has_country = any(country in response.lower() for country in ['canada', 'uk', 'france', 'taiwan', 'germany'])
    
    specificity_score = (has_number + has_statistic + has_country) - generic_count
    return specificity_score

def calculate_argument_engagement(response, previous_messages):
    # Does the response reference specific previous arguments?
    last_3_messages = previous_messages[-3:] if len(previous_messages) >= 3 else previous_messages
    
    engagement_count = 0
    for msg in last_3_messages:
        # Check if response mentions the speaker's name
        if msg.sender.lower() in response.lower():
            engagement_count += 1
    
    return engagement_count

def calculate_repetition_from_context(response, previous_messages):
    # Is the model just repeating what was already said?
    response_words = set(response.lower().split())
    
    all_previous_words = set()
    for msg in previous_messages:
        all_previous_words.update(msg.content.lower().split())
    
    # What fraction of response is NEW words not seen before?
    new_words = response_words - all_previous_words
    
    if len(response_words) == 0:
        return 0
    
    return len(new_words) / len(response_words)

def calculate_complexity_of_response(response):
    # Average sentence length (complex thoughts = longer sentences)
    sentences = response.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 0
    
    words_per_sentence = [len(s.split()) for s in sentences]
    return np.mean(words_per_sentence)

def calculate_emotional_engagement(response):
    # Strong emotional words indicate the model is still "in character"
    emotional_words = [
        'terrible', 'wonderful', 'crucial', 'devastating', 'outrageous',
        'unconscionable', 'heartbreaking', 'shameful', 'inspiring',
        'horrifying', 'absurd', 'ridiculous', 'shocking'
    ]
    
    response_lower = response.lower()
    return sum(1 for word in emotional_words if word in response_lower)

def analyze_response_comprehensive(response, previous_messages):
    metrics = {
        'length': len(response),
        'length_ratio': calculate_response_length_ratio(response, previous_messages),
        'specificity': calculate_response_specificity(response, previous_messages),
        'engagement': calculate_argument_engagement(response, previous_messages),
        'novelty': calculate_repetition_from_context(response, previous_messages),
        'complexity': calculate_complexity_of_response(response),
        'emotional': calculate_emotional_engagement(response),
    }
    return metrics



def extract_addressee(response):
    """
    Determines who Sarah is addressing in her response.
    Returns: 'Mike', 'Lisa', 'Tom', 'None', or 'Multiple'
    """
    response_lower = response.lower()
    
    mentions = {
        'Mike': 'mike' in response_lower,
        'Lisa': 'lisa' in response_lower,
        'Tom': 'tom' in response_lower
    }
    
    mentioned = [name for name, present in mentions.items() if present]
    
    if len(mentioned) == 0:
        return 'None'
    elif len(mentioned) == 1:
        return mentioned[0]
    else:
        return 'Multiple'

def get_last_speaker(messages):
    """Returns the name of the last speaker before Sarah responds"""
    if len(messages) == 0:
        return None
    return messages[-1].sender

def get_first_speaker(messages):
    """Returns the name of the first speaker"""
    if len(messages) == 0:
        return None
    return messages[0].sender

def get_second_speaker(messages):
    """Returns the name of the second speaker"""
    if len(messages) < 2:
        return None
    return messages[1].sender

def analyze_addressee_pattern(response, messages, names):
    """
    Analyzes who Sarah is addressing based on a list of participant names.
    
    Parameters:
    - response: Sarah's response text
    - messages: list of Message objects
    - names: list of participant names (e.g., ['Mike', 'Lisa', 'Tom'])
    
    Returns dictionary with addressee info and category
    """
    addressee = extract_addressee(response)
    
    # Build position mapping for each name
    positions = {}
    if len(messages) >= 1:
        positions['Last'] = messages[-1].sender
    if len(messages) >= 2:
        positions['Second-to-Last'] = messages[-2].sender
    if len(messages) >= 3:
        positions['Third-to-Last'] = messages[-3].sender
    if len(messages) >= 1:
        positions['First'] = messages[0].sender
    if len(messages) >= 2:
        positions['Second'] = messages[1].sender
    if len(messages) >= 3:
        positions['Third'] = messages[2].sender
    
    # Determine category
    if addressee == 'None':
        category = 'None'
    elif addressee == 'Multiple':
        category = 'Multiple'
    elif addressee not in names:
        category = 'Other'
    else:
        # Check which position this person holds
        # Priority: Last, Second-to-Last, Third-to-Last, then First, Second, Third
        category = 'Other'  # default if not found in key positions
        
        for position, speaker in positions.items():
            if addressee == speaker:
                category = position
                break
    
    return {
        'addressee': addressee,
        'category': category,
        'positions': positions,
    }