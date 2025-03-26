# Regular Expressions in NLP

Regular expressions (regex) are powerful patterns used to match and manipulate text. They are essential tools in natural language processing for tasks like tokenization, pattern matching, and text cleaning.

## Basic Regex Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| `.` | Matches any character except newline | `a.c` matches "abc", "adc", etc. |
| `^` | Matches start of a string | `^Hello` matches "Hello World" |
| `$` | Matches end of a string | `world$` matches "Hello world" |
| `*` | Matches 0 or more occurrences | `ab*c` matches "ac", "abc", "abbc", etc. |
| `+` | Matches 1 or more occurrences | `ab+c` matches "abc", "abbc", but not "ac" |
| `?` | Matches 0 or 1 occurrence | `ab?c` matches "ac" and "abc" |
| `\d` | Matches any digit | `\d\d\d` matches "123" |
| `\w` | Matches any alphanumeric character | `\w+` matches "hello_123" |
| `\s` | Matches any whitespace | `a\sb` matches "a b" |
| `[...]` | Matches any character in brackets | `[aeiou]` matches any vowel |
| `[^...]` | Matches any character not in brackets | `[^aeiou]` matches any consonant |

## Regex in Python with `re` Module

```python
import re

text = "Let's write RegEx! Won't that be fun? I sure think so. Can you find 4 sentences? Or perhaps, all 19 words?"

# Match sentence endings
sentence_endings = r"[.?!]"
sentences = re.split(sentence_endings, text)
print("Sentences:", sentences)

# Find all capitalized words
capitalized_words = r"[A-Z]\w+"
caps = re.findall(capitalized_words, text)
print("Capitalized words:", caps)

# Split on spaces
spaces = r"\s+"
words = re.split(spaces, text)
print("Words:", words)

# Find all digits
digits = r"\d+"
numbers = re.findall(digits, text)
print("Numbers:", numbers)
```

Output:
```
Sentences: ["Let's write RegEx", "  Won't that be fun", "  I sure think so", "  Can you find 4 sentences", "  Or perhaps, all 19 words", ""]
Capitalized words: ['Let', 'RegEx', 'Won', 'I', 'Can', 'Or']
Words: ["Let's", 'write', 'RegEx!', "Won't", 'that', 'be', 'fun?', 'I', 'sure', 'think', 'so.', 'Can', 'you', 'find', '4', 'sentences?', 'Or', 'perhaps,', 'all', '19', 'words?']
Numbers: ['4', '19']
```

## Match vs Search in Regex

- `match()`: Checks if the pattern matches at the **beginning** of the string
- `search()`: Checks if the pattern matches **anywhere** in the string

```python
import re

text = "Python is awesome"

# Using match
match_result = re.match(r"awesome", text)
print("Match result:", match_result)  # None, because "awesome" is not at the beginning

# Using search
search_result = re.search(r"awesome", text)
print("Search result:", search_result)  # <re.Match object; span=(10, 17), match='awesome'>
```

## Tokenization with NLTK and Regex

NLTK provides specialized tokenizers for different text types:

```python
# Import necessary modules
import re
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.tokenize import TweetTokenizer

# Example text
scene_one = "SCENE 1: An empty street. A KING enters, followed by a SERVANT."

# Sentence tokenization
sentences = sent_tokenize(scene_one)
print("Sentences:", sentences)

# Word tokenization
tokenized_sent = word_tokenize(sentences[0])
print("Words in first sentence:", tokenized_sent)

# Get unique tokens
unique_tokens = set(word_tokenize(scene_one))
print("Unique tokens:", unique_tokens)

# Using regex tokenization for specific patterns
tweets = ["I #love #NLP and #Python!", "@user replied: #NLProc is amazing!"]

# Define a pattern to find hashtags
pattern1 = r"#\w+"
hashtags = regexp_tokenize(tweets[0], pattern1)
print("Hashtags:", hashtags)

# Define a pattern for both mentions and hashtags
pattern2 = r"([@#]\w+)"
mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
print("Mentions and hashtags:", mentions_hashtags)

# Using TweetTokenizer for Twitter data
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print("Tweet tokens:", all_tokens)
```

## Special Use Cases

### Working with Different Languages

```python
german_text = "Ich hätte gerne Üben mit Regex. Kleine Übungen sind 😊 für Anfänger."

# Tokenize and print only capital words (including German letters)
capital_words = r"[A-ZÜ]\w+"
print("Capital words:", regexp_tokenize(german_text, capital_words))

# Tokenize and print only emoji
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print("Emoji:", regexp_tokenize(german_text, emoji))
```

### Analyzing Script Structure

```python
import re
import matplotlib.pyplot as plt
from nltk.tokenize import regexp_tokenize

# Script text (example)
holy_grail = """
KING ARTHUR: Knights of Ni, we are but simple travelers who seek the enchanter who lives beyond these woods.
KNIGHT: Ni! Ni! Ni! Ni!
KING ARTHUR: Oh, what sad times are these when passing ruffians can say Ni at will to old ladies.
SIR BEDEVERE: We are now the Knights Who Say Ecky-ecky-ecky-ecky-pikang-zoop-boing-goodem-zoo-owli-zhiv.
"""

# Split the script into lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = r"[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line
tokenized_lines = [regexp_tokenize(s, r"\w+") for s in lines]

# Count words per line
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)
plt.title("Distribution of Line Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()
```

## Tips for Effective Regex in NLP

1. **Start Simple**: Begin with basic patterns and incrementally add complexity
2. **Test Thoroughly**: Use tools like regex101.com to test your patterns
3. **Consider Edge Cases**: Account for special characters, different languages, etc.
4. **Use Raw Strings**: In Python, prefix regex patterns with `r` to avoid escape character issues
5. **Balance Precision**: Too specific patterns may miss valid matches; too general may include false positives 