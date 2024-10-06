# Quest 2: Ghost Result Mitigation in Speech-to-Text

The script defines a framework for two types of speech-to-text (STT) models: a basic model (`BasicSpeechToText`) and an advanced model (`GhostReductionSpeechToText`) designed specifically to reduce ghost results in transcriptions.

This documentation provides a conceptual overview of the components of the script and highlights the modifications implemented to mitigate ghost results.

## Components Overview

1. **BaseSpeechToText Class**:
   - An abstract base class providing a common structure for different STT models.
   - Initializes the device (CPU/GPU), loads a pretrained Wav2Vec2 model, and defines an abstract method for generating transcriptions based on audio input.
   - **Key Method**: `transcribe_audio` processes the audio input through the model to produce a transcription.

2. **BasicSpeechToText Class**:
   - Inherits from `BaseSpeechToText`.
   - Implements the `generate_response` method, which calls the `load_audio` and `transcribe_audio` methods to provide a transcription of the audio file.
   - **Contribution**: Serves as a baseline model for comparison against the advanced ghost reduction model.

3. **GhostReductionSpeechToText Class**:
   - Inherits from `BaseSpeechToText` and incorporates additional components aimed at reducing ghost results.
   - **Key Components**:
     - **Noise Reduction**: Implements audio filtering to minimize background noise, enhancing the clarity of the speech input.
     - **Confidence Scoring**: Calculates confidence scores for predicted transcriptions to assess their reliability.
     - **Language Model Corrections**: Applies a dictionary of common misinterpretations to correct potential errors based on context.

4. **Methods of GhostReductionSpeechToText**:
   - **generate_response**: Augments the basic transcription process by adding noise reduction, confidence scoring, and language model corrections.
   - **reduce_noise**: Applies a bandpass filter to improve the audio quality by focusing on relevant frequency ranges for human speech.
   - **enhance_acoustic_model**: Implements pre-emphasis techniques to enhance the signal quality of the audio.
   - **transcribe_audio_with_confidence**: Generates transcriptions alongside confidence scores, facilitating the assessment of output reliability.
   - **apply_language_model**: Corrects transcriptions based on predefined common replacements, ensuring contextually appropriate outputs.
   - **filter_low_confidence**: Eliminates words from the final transcription that fall below a specified confidence threshold.

## Changes Implemented for Ghost Result Mitigation

### 1. Noise Reduction
This involves improving audio input quality to minimize distractions and enhance transcription accuracy.

- **Bandpass Filtering**:
   - The `reduce_noise` method applies a bandpass filter, specifically tuned to human speech frequencies (300 Hz to 3400 Hz). This approach significantly cleans the audio input.
   - **Contribution**: Reduces the potential for ghost results caused by ambient noise, allowing the model to focus on the speech content.

### 2. Confidence Scoring
Integrating a confidence scoring mechanism to evaluate the reliability of transcriptions post-processing.

- **Softmax Probability Analysis**:
   - The `transcribe_audio_with_confidence` method computes softmax probabilities to derive confidence scores for each predicted word.
   - **Contribution**: Ensures that only high-confidence transcriptions are included in the final output, reducing the likelihood of ghost results from uncertain predictions.

### 3. Language Model Corrections
The application of contextual knowledge to refine transcriptions, ensuring that errors stemming from misinterpretation are minimized.

- **Common Replacements Dictionary**:
   - The `apply_language_model` method utilizes a predefined dictionary of common misinterpretations to replace words that may have been incorrectly transcribed.
   - **Contribution**: Enhances the contextual accuracy of transcriptions, reducing the chance of ghost results caused by phonetic confusion.

### 4. Output Filtering
Implementing a filtering mechanism to retain only the most reliable parts of the transcription.

- **Confidence Thresholding**:
   - The `filter_low_confidence` method systematically removes words that do not meet the confidence threshold, ensuring that the final output is based on reliable predictions.
   - **Contribution**: Directly addresses ghost results by ensuring that low-confidence words do not contaminate the transcription.
