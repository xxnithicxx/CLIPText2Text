from underthesea import word_tokenize
import random

def contaminate_text(sentence, contamination_prob=0.3):
    """
    Randomly removes diacritics from words in a Vietnamese sentence.
    
    Args:
        sentence (str): Input Vietnamese sentence.
        contamination_prob (float): Probability of removing diacritics from a word.
    
    Returns:
        str: Contaminated sentence.
        list: List of tuples (word, is_contaminated) for ground truth labels.
    """
    def remove_diacritics(word):
        DIACRITICS_MAP = str.maketrans(
            "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ",
            "aaaaaăăăăăâââââeeeeeêêêêêiiiiiôôôôôôơơơơơuuuuuưưưưưyyyyyd"
        )
        return word.translate(DIACRITICS_MAP)

    # Tokenize the sentence
    tokenized_words = word_tokenize(sentence)
    
    contaminated_sentence = []
    labels = []

    for word in tokenized_words:
        # Randomly decide whether to contaminate the word
        if random.random() < contamination_prob:
            contaminated_word = remove_diacritics(word)
            contaminated_sentence.append(contaminated_word)
            labels.append((word, 1))  # 1 indicates contamination
        else:
            contaminated_sentence.append(word)
            labels.append((word, 0))  # 0 indicates no contamination

    return " ".join(contaminated_sentence), labels

# Example input
sentence = "Xin chào, tôi là Thịnh và tôi đang học lập trình."
contaminated_sentence, labels = contaminate_text(sentence)

print("Original Sentence:", sentence)
print("Contaminated Sentence:", contaminated_sentence)
print("Labels:", labels)
