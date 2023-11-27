# Changelog

<!--next-version-placeholder-->


## v1.3.0 (27/11/2023)

 - refactor: improve pdf reader
 - fix: pypdf dependency

## v1.2.0 (24/11/2023)

 - fix: convert dataframe to json

## v1.1.0 (23/11/2023)

 - feat: add support to async.
 - Major modification: Conversation was named as Chatter and QuestionContext doesn't handle anymore the context: use Context class's instance to generate a context text. This behavior is temporally and will be incorporate back to QuestionContext in further releases.
 - Add text processors and extractor (pdf only)
 - Implement semantic segmentation

## v0.4.0 (31/10/2023)

 - feat(BaseContext): add embeddings as json (dict)
 - feat: add option to send only the current message instead of the entire conversation in a 'answer' call

## v0.3.1 (25/10/2023)
 - fix: missing enumerate at base._reduce_number_of_tokens_if_needed
 - feat: add option to set a chatgpt agent to summarize converation if exceed tokens
 - fix(base): handle error when trying to generate embeddings

## v0.2.10 (24/10/2023)
 - docs(QuestionContext): pass Opan Ai's APIError message and code as answer
 - docs: change openai's error message
 - fix(QuestionContext): handle openai's bad gateway error
 - fix: remove unnecessary logger setting (file)
 - fix: python and scipy compatibility
 - update: add .env310 to .gitignore
## v0.2.3 (24/08/2023)

- update: add constants to represent embeddings table columns

## v0.2.2 (24/08/2023)

- update: add class QuestionContext to init.

## v0.1.0 (12/07/2023)

- First release of `genaikit`!