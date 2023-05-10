import math
from collections import defaultdict
import random


def n_grams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class KneserNeyInterpolated:
    def __init__(self, order, discount=0.75):
        self.order = order
        self.discount = discount
        self.ngram_counts = [defaultdict(int) for _ in range(order)]
        self.context_counts = [defaultdict(int) for _ in range(order - 1)]
        self.vocab = set()

    def train(self, tokens):
        self.vocab.update(tokens)
        for n in range(1, self.order + 1):
            for ngram in n_grams(tokens, n):
                self.ngram_counts[n - 1][ngram] += 1
                if n > 1:
                    self.context_counts[n - 2][ngram[:-1]] += 1

    def kneser_ney_weight(self, ngram):
        order = len(ngram)
        count = self.ngram_counts[order - 1][ngram]
        if count == 0:
            return 0
        context_count = self.context_counts[order - 2][ngram[:-1]] if order > 1 else len(self.vocab)
        discount = self.discount
        return max(count - discount, 0) / context_count 
    
    def continuation_prob(self, ngram):
        order = len(ngram)
        if order == 1:
            return 1 / len(self.vocab)
        context = ngram[:-1]
        continuation_count = len([ngram for ngram in self.ngram_counts[order - 2] if ngram[:-1] == context])
        context_count = self.context_counts[order - 2][context] if order > 2 else len(self.vocab)
        return continuation_count / context_count if context_count > 0 else 0
    
    def prob(self, ngram):
        if len(ngram) > self.order:
            raise ValueError("ngram order exceeds model order")
        if len(ngram) == 1:
            return self.continuation_prob(ngram)
        else:
            return self.kneser_ney_weight(ngram) + self.continuation_prob(ngram) * self.prob(ngram[1:])
        
    def logscore(self, ngram):
        return math.log(self.prob(ngram))
    
    def generate(self, context=None, max_length=100, seed=None):
        if seed is not None:
            random.seed(seed)
        if context is None:
            context = []
        elif len(context) >= self.order - 1:
            context = context[-(self.order - 1):]
        
        generated_tokens = list(context)

        while len(generated_tokens) < max_length:
            if len(generated_tokens) < self.order - 1:
                ngram = tuple(generated_tokens)
            else:
                ngram = tuple(generated_tokens[-(self.order - 1):])
            
            candidates = [(ngram + (token,), self.prob(ngram + (token,))) for token in self.vocab]
            candidates.sort(key=lambda x: x[1], reverse=True)

            if candidates:
                best_candidate = candidates[0][0]
                generated_tokens.append(best_candidate[-1])
            else:
                break

        return generated_tokens