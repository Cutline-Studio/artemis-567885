# Artemis corpus manifest

Source: Project Gutenberg plain-text ebooks (https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt), downloaded 2026-10-23.
License: these works are in the public domain in the United States. The Project Gutenberg header and footer (license boilerplate outside the '*** START OF' / '*** END OF' markers) were removed; per the Project Gutenberg license, the stripped texts carry no Project Gutenberg trademark. Status outside the US was not checked.
Split is at book level with distinct authors per split (train: Dickens, Tolstoy/Maude translation, Melville, Hugo; val: Austen; test: Twain). Builder: build_corpus.py. Bytes are UTF-8 file sizes after stripping.

| split | id | title | author | bytes | sha256 |
|---|---|---|---|---|---|
| train | 1023 | Bleak House | Charles Dickens | 2024933 | f5a63bc64af664bb555a3cf8c2804e1d2b612cc07d898d567a1d5aa8e83edd85 |
| train | 766 | David Copperfield | Charles Dickens | 2013494 | dcc5a23ccdf78e8fa00329950af99270ce10e7801d183cd8f59966b7252722aa |
| train | 2600 | War and Peace | graf Leo Tolstoy | 3339773 | 841254e45cac8dfda9ff0870f544e67eff7302e9bf56614c5efe5f4a9b6bd479 |
| train | 2701 | Moby Dick; Or, The Whale | Herman Melville | 1256486 | 9c05ad4f019bc4f8d1462bef1e2b3c0a84ff3d13482bbe877b92659c0c8e00a8 |
| train | 135 | Les Misérables | Victor Hugo | 3349400 | d005a0c4b44d6a2cefc5353ae2adc6a11dd47aadcd4015c9cae8ad8046867fcd |
| val | 1342 | Pride and Prejudice | Jane Austen | 752442 | 3709883de831a8cd1b6f2ead5640644c4e26abe972ca382b774b221c90c26bd5 |
| val | 105 | Persuasion | Jane Austen | 477763 | 6b27ade16e38576309ab7b594c0d96e0281760c8cabdfb3ee02ae6f8c559a7e8 |
| test | 76 | Adventures of Huckleberry Finn | Mark Twain | 602618 | c00901b5e87adb23b0008c70bbb11304b74abccad33ad4631b2b6b779271ce95 |
| test | 74 | The Adventures of Tom Sawyer, Complete | Mark Twain | 414509 | 916902210dff1b2818fbafa3a6239dc685129b7a08334312873015e88a1aed4a |

Totals: train 11,984,086 bytes, val 1,230,205 bytes, test 1,017,127 bytes
