from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def rouge_l_score(reference_words, hypothesis_words):
    # reference_words = reference.split()
    # hypothesis_words = hypothesis.split()

    lcs = [[0] * (len(hypothesis_words) + 1) for _ in range(len(reference_words) + 1)]
    for i in range(len(reference_words) + 1):
        for j in range(len(hypothesis_words) + 1):
            if i == 0 or j == 0:
                lcs[i][j] = 0
            elif reference_words[i - 1] == hypothesis_words[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    return lcs[-1][-1] / len(reference_words)
