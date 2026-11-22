# Hugging Face notebook

[`xlsr_gpt.ipynb`](xlsr_gpt.ipynb) demonstrates experimental decoding for the
original Taiwanese Wav2Vec2/GPT-2 pairing. It is not a drop-in recipe for
English checkpoints.

## Using English Wav2Vec2 with GPT-Neo or BERT

The notebook multiplies an acoustic score and a language-model score at each
position. That only makes sense if the scores already refer to the same
positions and the same symbol at every token ID. English
[Wav2Vec2 Base 960h](https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/vocab.json)
has 32 CTC symbols, including a blank and a word separator.
[GPT-Neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125m/raw/main/config.json)
has 50,257 output IDs; [BERT Base Uncased](https://huggingface.co/google-bert/bert-base-uncased/raw/main/config.json)
has 30,522. Their token IDs also mean different things, so slicing or padding
the tensors does not fix the mismatch.

1. Start with the [Wav2Vec2 CTC example](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
   using the English checkpoint and its own processor. The greedy baseline is
   `processor.batch_decode(logits.argmax(dim=-1))`.
2. For rescoring, use a CTC beam decoder to get several candidate **text
   transcriptions** and their acoustic log scores. For example,
   [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) accepts the
   unpadded per-utterance logits and labels in acoustic token-ID order; keep the
   checkpoint's CTC blank and word separator.
3. Tokenize each candidate with the language model's **own** tokenizer.
   For GPT-Neo, compute causal token log probabilities (prepend its BOS token
   to score the first candidate token). For BERT, use `AutoModelForMaskedLM`
   and mask one non-special token at a time; score the original token at the
   masked position. The latter is a pseudo-log-likelihood, not a causal
   next-token score.
4. Rank candidates with a heuristic such as
   `acoustic_log_score + weight * mean_lm_token_log_score`. Set the weight
   using development-set WER and compare against acoustic-only decoding.
   Rescoring can only choose among the candidates the CTC decoder kept.

If a lightweight English LM baseline is enough, see the
[Wav2Vec2 n-gram decoding guide](https://huggingface.co/blog/wav2vec2-with-ngram).
