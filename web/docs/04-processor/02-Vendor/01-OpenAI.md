---
Title: OpenAI LLM Processor
sidebar_label: OpenAI
---

This page contains information about all the OpenAI processors that are available in LLMStack.

### Prerequisites
In order to use the OpenAI processors, you need to have an OpenAI API key configured in the settings page (http://localhost:8000/settings). You can get the API key from [here](https://platform.openai.com/).

### Processors
| Processor | Description | Input Parameters | Output | Configuration | Docs |
|---|---|---|---|---|---|
| Chat Completions | Allows you to interact with the OpenAI's ChatGPT API | System message, List of user messages | Returns a list of message completions | Controls the behavior of the assistant | [Link](https://platform.openai.com/docs/guides/chat) |
| Completions API | Allows you to interact with the OpenAI's Completions API | Prompt | Returns a list of completions the model predicted for the given prompt input | Controls the behavior of the model | [Link](https://platform.openai.com/docs/guides/completion) |
| Image Generation (DallE) | Allows you to interact with the OpenAI's Image Creation API (DallE) | Text prompt | Returns a list of images generated by the model | Controls the behavior of the model | [Link](https://platform.openai.com/docs/guides/images) |
| Image Edit (Dall-E) | Allows you to interact with the OpenAI's Image Edit API (Dall-E) | Prompt, Image, Optional mask image | Returns a list of images generated by the model | Controls the behavior of the model | [Link](https://platform.openai.com/docs/guides/images) |
| Image Variation (Dall-E) | Allows you to interact with the OpenAI's Image Variation API (Dall-E) | Image for variation | Returns a list of images generated by the model | Controls the behavior of the model | [Link](https://platform.openai.com/docs/guides/images) |
| Audio Transcription (Whisper) | Allows you to interact with the OpenAI's Transcription Whisper API | Audio file (mp3, mp4, mpeg, mpga, m4a, wav, or webm) | Returns a text, which is the transcription of the audio file | Configuration parameters like language, response format, etc. | [Link](https://platform.openai.com/docs/guides/audio) |
| Audio Translation (Whisper) | Allows you to interact with the OpenAI's Translation Whisper API | Audio file (mp3, mp4, mpeg, mpga, m4a, wav, or webm) | Returns a text, which is the translation of the audio file | Configuration parameters like model, response format etc. | [Link](https://platform.openai.com/docs/guides/audio) |