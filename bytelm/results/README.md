# Corpus benchmark results (measured, CPU, no cloud charges)

Command: `python3 artemis_corpus_benchmark.py --data corpus --output runs/seedS --seed S --epochs 10 --steps 1000` for S in 42, 1, 2.
Raw unedited consoles: `seed*_console.txt`. Environment: `env.txt`. Checkpoints (.pt) not committed; SHA-256 is in each console.

| Seed | Val BPB model / unigram | Test BPB model / unigram | Wall s |
|---|---|---|---|
| 42 | 3.6192 / 4.5817 | 3.5860 / 4.7079 | 30.1 |
| 1 | 3.6092 / 4.5817 | 3.5535 / 4.7079 | 29.2 |
| 2 | 3.6147 / 4.5817 | 3.5551 / 4.7079 | 27.0 |

1,000 steps (~2.05 MB of targets) is under one pass of the 12 MB train split, so only one post-training validation checkpoint (epoch 1) existed; selection was trivial. Test scored 1,017,125 bytes.
