# Shout
Post ASR Morphological Reconstruction for Polysynthetic Languages

## Why? 
Many STT models fail at the "boundaries" of words in polysynthetic languages such as: Cree, Mohawk, or Blackfoot. As shown by research and projects from the FLAIR group.
The idea here is to use a base STT model to output garbled chunks and feed it's output to a browser based "Morphological Adapter"

## How? (Maybe)
I will try and train a small LoRA for a local model based on the grammatical rules of the target language/dialect.
The adapter should ideally "fix" the broken STT output in real-time within the browser. 
This way for lower support languages we get a voice to text that goes from phonetics - > semantics
