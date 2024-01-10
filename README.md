This script takes all the pdf and txt files in a folder, splits them in 500 characters and transform each chunks in vectors using the openAi embedding.

A question is asked and the 4 most relevant chunks are send as a context to the openAI API as a context to answer the question.
