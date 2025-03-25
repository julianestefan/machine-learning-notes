# Working with Images and Audio in Hugging Face

Hugging Face provides powerful pipelines for processing and analyzing both image and audio data. This guide covers preprocessing techniques and various tasks for these modalities.

## Image Processing with Hugging Face

Images in machine learning are represented as arrays of pixel values. Before feeding images to models, preprocessing is crucial to:

- Meet model input specifications
- Maintain consistency across datasets
- Focus on relevant image regions
- Improve model performance

### Image Preprocessing Techniques

#### 1. Cropping

Cropping removes parts of an image to focus on the most relevant regions or to standardize dimensions.

```python 
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as image_transforms

# Load an example image
original_image = Image.open("example_image.jpg")

# Create the numpy array
image_array = np.array(original_image)

# Crop the center of the image
cropped_image = image_transforms.CenterCrop(size=(200, 200))(original_image)

# Display the result
imgplot = plt.imshow(cropped_image)
plt.show()
```

#### 2. Resizing

Resizing changes the width and height of an image, often to match model input requirements.

```python
# Import necessary libraries
from PIL import Image
from torchvision import transforms

# Load an image
image = Image.open("example_image.jpg")

# Resize the image
resized_image = transforms.Resize((224, 224))(image)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(resized_image)
plt.title("Resized Image (224x224)")
plt.show()
```

### Image Classification

Image classification identifies what an image contains from a predefined set of categories.

```python 
# Import necessary libraries
from transformers import pipeline
import matplotlib.pyplot as plt
from PIL import Image

# Load an image
image = Image.open("example_image.jpg")

# Create the pipeline
image_classifier = pipeline(task="image-classification", 
                        model="abhishek/autotrain_fashion_mnist_vit_base")

# Predict the class of the image
results = image_classifier(image)

# Print the results
print(f"Top prediction: {results[0]['label']} with confidence {results[0]['score']:.4f}")

# Display image with prediction
plt.imshow(image)
plt.title(f"Predicted: {results[0]['label']}")
plt.axis('off')
plt.show()
```

### Question Answering on Images

Hugging Face supports two types of image-based question answering:

#### 1. Document Question Answering

This task extracts answers to questions from documents (forms, invoices, receipts, etc.).

```python 
# Import necessary libraries
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

# Create the pipeline
dqa = pipeline(task="document-question-answering", 
              model="naver-clova-ix/donut-base-finetuned-docvqa")

# Load the document image
image = Image.open("document.png")

# Set the question
question = "Which meeting is this document about?"

# Get the answer
results = dqa(image=image, question=question)

# Display results
print(f"Question: {question}")
print(f"Answer: {results[0]['answer']} (confidence: {results[0]['score']:.4f})")

# Show the document
plt.figure(figsize=(10, 14))
plt.imshow(image)
plt.title("Document Image")
plt.axis('off')
plt.show()
```

Output: 
```shell
Question: Which meeting is this document about?
Answer: takeda global risk management forum (confidence: 0.7789)
```

#### 2. Visual Question Answering

This task answers general questions about the content of images.

```python 
# Import necessary libraries
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

# Load an image
image = Image.open("example_image.jpg")

# Create pipeline
vqa = pipeline(task="visual-question-answering", 
              model="dandelin/vilt-b32-finetuned-vqa")

# Set the question
question = "What is the person wearing on their head?"

# Use image and question in vqa
results = vqa(image=image, question=question)

# Show top 3 answers
print(f"Question: {question}")
for i, result in enumerate(results[:3]):
    print(f"Answer {i+1}: {result['answer']} (confidence: {result['score']:.4f})")

# Display the image
plt.imshow(image)
plt.title(f"Q: {question} | A: {results[0]['answer']}")
plt.axis('off')
plt.show()
```

Output: 
```shell 
Question: What is the person wearing on their head?
Answer 1: hat (confidence: 0.9796)
Answer 2: beanie (confidence: 0.5232)
Answer 3: cap (confidence: 0.2478)
```

## Audio Processing with Hugging Face

Audio in machine learning is typically represented as waveforms with amplitude values over time. Key considerations when working with audio include:

### Sample Rate

The sample rate is the number of audio samples recorded per second, measured in Hertz (Hz). Common sample rates include:

- 16,000 Hz for speech processing
- 44,100 Hz for music (CD quality)
- 48,000 Hz for professional audio

### Audio Preprocessing Techniques

#### 1. Resampling

Resampling aligns sample rates across all files to ensure consistency and compatibility with models.

```python 
# Import necessary libraries
from datasets import load_dataset, Audio

# Load an audio dataset
audio_file = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:10]")

# Save the old sampling rate
old_sampling_rate = audio_file[1]["audio"]["sampling_rate"]

# Resample the audio files
audio_file = audio_file.cast_column("audio", Audio(sampling_rate=16_000))

# Compare the old and new sampling rates
print("Old sampling rate:", old_sampling_rate)
print("New sampling rate:", audio_file[1]["audio"]["sampling_rate"])
```

#### 2. Filtering by Duration

Filtering by duration ensures data quality and reduces computational requirements.

```python 
# Import necessary libraries
import librosa
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:100]")

# Create a list of durations
old_durations_list = []

# Loop over dataset
for row in dataset['path']:
    old_durations_list.append(librosa.get_duration(path=row))

# Create a new column
dataset = dataset.add_column("duration", old_durations_list)

# Filter the dataset to keep only files shorter than 6 seconds
filtered_dataset = dataset.filter(lambda d: d < 6.0, input_columns=["duration"], keep_in_memory=True)

# Print statistics
print(f"Original dataset size: {len(dataset)}")
print(f"Filtered dataset size: {len(filtered_dataset)}")
print(f"Min duration: {min(filtered_dataset['duration']):.2f}s")
print(f"Max duration: {max(filtered_dataset['duration']):.2f}s")
```

### Audio Classification

Audio classification identifies the category of an audio sample, such as language, speaker, or environmental sound.

```python 
# Import necessary libraries
from transformers import pipeline
from datasets import load_dataset

# Load a multilingual dataset
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "de", split="validation[:10]")

# Create the pipeline for language identification
classifier = pipeline(task="audio-classification", model="facebook/mms-lid-126")

# Extract the sample
audio = dataset[1]["audio"]["array"]
sentence = dataset[1]["sentence"]

# Predict the language
prediction = classifier(audio)

print(f"Predicted language is '{prediction[0]['label'].upper()}' for the sentence '{sentence}'")
print(f"Top 3 language predictions:")
for i, pred in enumerate(prediction[:3]):
    print(f"{i+1}. {pred['label'].upper()}: {pred['score']:.4f}")
```

Output:
```shell
Predicted language is 'DEU' for the sentence 'Deswegen ballert es mehr.'
Top 3 language predictions:
1. DEU: 0.9978
2. AUT: 0.0012
3. NOR: 0.0003
```

### Automatic Speech Recognition (ASR)

ASR converts spoken language into text. This technology powers voice assistants, transcription services, and more.

```python 
# Import necessary libraries
from transformers import pipeline
from datasets import load_dataset
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Load a dataset
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:5]")
example = dataset[0]

# Create an ASR pipeline using Meta's wav2vec model
meta_asr = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# Predict the text from the example audio
meta_pred = meta_asr(example["audio"]["array"])["text"].lower()

# Repeat for OpenAI's Whisper model
open_asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-tiny")
open_pred = open_asr(example["audio"]["array"])["text"].lower()

# Print the prediction from both models
print("Original sentence:", example["sentence"].lower())
print("META ASR result:", meta_pred)
print("OpenAI Whisper result:", open_pred)

# Visualize the audio waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(example["audio"]["array"], sr=example["audio"]["sampling_rate"])
plt.title("Audio Waveform")
plt.tight_layout()
plt.show()
```

Output:
```shell
Original sentence: it is a charity school whose fees are calculated on a means test.
META ASR result: it is a charity school whose feeds are calculated on a men test
OPENAI ASR result: it is a charity school whose fees are calculated on a means test.
```

### Evaluating ASR Models

Word Error Rate (WER) is the standard metric for evaluating ASR systems. It's based on the Levenshtein distance, which measures the minimum number of single-character edits required to change one string into another.

WER ranges from 0 to infinity, where:
- 0 represents perfect transcription
- Values closer to 0 indicate better performance

```python 
# Import necessary libraries
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import evaluate

# Load dataset and models
english = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:3]")
meta_asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
open_asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Load the WER metric
wer = evaluate.load("wer")

# Create the data function
def data(n=3):
    for i in range(n):
        yield english[i]["audio"]["array"], english[i]["sentence"].lower()
        
# Predict the text for the audio file with both models
output = []
for audio, sentence in data():
    meta_pred = meta_asr(audio)["text"].lower()
    open_pred = open_asr(audio)["text"].lower()
    # Append to output list
    output.append({"sentence": sentence, "metaPred": meta_pred, "openPred": open_pred})

# Create a dataframe for better visualization
output_df = pd.DataFrame(output)

# Display the results
print("Reference vs. Predictions:")
for i, row in output_df.iterrows():
    print(f"\nSample {i+1}:")
    print(f"Reference: {row['sentence']}")
    print(f"Meta ASR: {row['metaPred']}")
    print(f"OpenAI: {row['openPred']}")

# Compute the WER for both models
metaWER = wer.compute(predictions=output_df["metaPred"], references=output_df["sentence"])
openWER = wer.compute(predictions=output_df["openPred"], references=output_df["sentence"])

# Print the WER
print(f"\nPerformance Comparison:")
print(f"Meta ASR WER: {metaWER:.4f} (lower is better)")
print(f"OpenAI Whisper WER: {openWER:.4f} (lower is better)")
print(f"Relative improvement of Whisper over Meta: {((metaWER - openWER) / metaWER) * 100:.1f}%")
```

Output:
```shell
Reference vs. Predictions:

Sample 1:
Reference: it is a charity school whose fees are calculated on a means test.
Meta ASR: it is a charity school whose feeds are calculated on a men test
OpenAI: it is a charity school whose fees are calculated on a means test.

Sample 2:
Reference: each cottage has a unique style.
Meta ASR: each cut has a unique style
OpenAI: each cottage has a unique style.

Sample 3:
Reference: they had danced all night.
Meta ASR: they had danced all night
OpenAI: they had danced all night.

Performance Comparison:
Meta ASR WER: 0.6098 (lower is better)
OpenAI Whisper WER: 0.2439 (lower is better)
Relative improvement of Whisper over Meta: 60.0%
```

## Conclusion

Hugging Face provides powerful and accessible tools for working with both image and audio data. By combining these pipelines with proper preprocessing techniques, you can build sophisticated applications for tasks such as image classification, document analysis, speech recognition, and more.

The ease of use and consistent API design make it simple to experiment with different models and approaches without needing to understand the complex architecture details of each model.