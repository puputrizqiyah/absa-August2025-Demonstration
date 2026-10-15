# Zero-shot comparator: prompt and decoding configuration

Reproduced from Appendix A of the manuscript. The literal request string is kept
here so the pre-annotation run can be repeated without reverse-engineering the
annotation guideline.

Decoding configuration: model snapshot `gpt-4o-mini-2024-07-18`, temperature 0, one
request per batch of ten records, JSON output enforced by a response schema, at most
4,096 output tokens, no retries. A response that fails to parse is recorded as an
absent label rather than silently repaired, which is why the coverage of the two
conditional tasks differs and why the coverage-aware and conditional scores of
Table 6 are not interchangeable.

## System instruction

You are an expert annotation assistant for Aspect-Based Sentiment Analysis on
Indonesian tweets about the August 2025 demonstrations in Indonesia.

Assign:

1. relevance: true if the tweet addresses the demonstrations, explicitly or
   implicitly, and false for promotion, spam, or unrelated content, in which case
   leave the aspect and polarity empty;
2. the single dominant aspect among Pemerintah (government, parliament,
   presidency, and other non-police state actors), Polisi (the police apparatus),
   and Rakyat (civil society and non-state actors);
3. the polarity toward that aspect, one of Negatif, Netral, or Positif.

If several aspects or polarities co-occur, label the most dominant or most
emotionally emphasised one and ignore the minor ones. Reply with a JSON array, one
object per input, in input order, with the fields index, relevant, aspect, and
label.

## User instruction

Annotate the tweets below according to the guideline.

[index] Text: ...
[index] Text: ...

(batches of ten records, in corpus order)

## Parsing rule

The JSON array is read positionally against the input batch and the aspect and
polarity fields are mapped to the study's three-by-three label set. A record whose
relevance field is false, or whose aspect field is empty, yields a missing value
for the conditional tasks rather than an imputed one.
