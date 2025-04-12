import random
import matplotlib.pyplot as plt
import argparse
from collections import Counter

vocab = ['A', 'B', 'C', 'D']

# conditional distributions for p(x_t | x_{<t})
# Format: given string -> probability distribution over following tokens
p_conditional = {
    '': {'A': 0.1, 'B': 0.3, 'C': 0.4, 'D': 0.2},
    'A': {'A': 0.2, 'B': 0.2, 'C': 0.3, 'D': 0.3},
    'B': {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2},
    'C': {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
    'D': {'A': 0.1, 'B': 0.4, 'C': 0.2, 'D': 0.3}
}

# draft model conditional distribution
q_conditional = {
    ctx: {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
    for ctx in p_conditional
}

def sample_from_dist(dist):
    items = list(dist.items())
    tokens, probs = zip(*items)
    return random.choices(tokens, weights=probs, k=1)[0]

def normalize(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}

def correction_distribution(p, q):
    beta = sum(min(p[token], q[token]) for token in p)
    correction = {}

    for token in p:
        leftover = p[token] - min(p[token], q[token])
        correction[token] = leftover / (1 - beta) if (1 - beta) > 0 else 0.0
    return correction

def speculative_decode_sequence(p_model, q_model, num_sequences, draft_block_size=4):
    samples = []

    for _ in range(num_sequences):
        context = ''
        sequence = []

        while len(sequence) < draft_block_size:
            # n-gram proposal - draft model generates a candidate sequence
            draft_tokens = []
            for _ in range(draft_block_size - len(sequence)):
                q_t = q_model[context]
                token = sample_from_dist(q_t)
                draft_tokens.append(token)
                context = token

            # verification step - target model verifies draft model's sequence
            context = ''  # reset to true model context
            accepted = []
            for token in draft_tokens:
                p_t = p_model[context]
                q_t = q_model[context]
                accept_prob = min(1.0, p_t[token] / q_t[token])
                if random.random() < accept_prob:
                    accepted.append(token)
                    context = token
                else:
                    break

            sequence.extend(accepted)

            # if rejected, use correction distribution to sample remaining tokens
            if len(accepted) < len(draft_tokens):
                correction_dist = correction_distribution(p_model[context], q_model[context])
                token = sample_from_dist(correction_dist)
                sequence.append(token)
                context = token

        samples.append(''.join(sequence))

    return samples

def run_test(num_samples=10000, graph=False):
    samples = speculative_decode_sequence(p_conditional, q_conditional, num_samples)
    first_token_counts = Counter(seq[0] for seq in samples)
    first_token_dist = normalize(first_token_counts)

    print("True First-Token Distribution (p):")
    for token in vocab:
        print(f"  {token}: {p_conditional[''][token]:.3f}")
    print("\nEmpirical First-Token Distribution (from speculative sampling):")
    for token in vocab:
        print(f"  {token}: {first_token_dist.get(token, 0):.3f}")

    if graph:
        plt.bar(vocab, [p_conditional[''][t] for t in vocab], width=0.4, label='True p(x₁)', align='edge')
        plt.bar(vocab, [first_token_dist.get(t, 0) for t in vocab], width=-0.4, label='Empirical', align='edge')
        plt.title("Speculative Sampling First Token vs Target Model")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speculative sequence decoding simulation')
    parser.add_argument('--graph', action='store_true', help='Display graph of distributions')
    args = parser.parse_args()

    run_test(graph=args.graph)
