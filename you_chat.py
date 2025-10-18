from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()
##indexing
# doc loading
video_id='kGlvBiFD7FU'
try:

    transcript_list= YouTubeTranscriptApi().fetch(video_id,languages=['hi'])

    transcript_preprocesssed=transcript_list.to_raw_data()

    transcript=" ".join(chunk['text'] for chunk in transcript_preprocesssed)
    print(transcript)
except TranscriptsDisabled:
    print("No Caption available for the video!!")

except Exception as e:
    print('an error occured {e}')

# text splitting

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=550,
#     chunk_overlap=150
# )
# chunks=splitter.create_documents([transcript])
# print(chunks[2].page_content)
# print(len(chunks))


# embeddings

# embeddings=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
# vector_store=FAISS.from_documents(chunks,embeddings)
# print(vector_store.index_to_docstore_id)

# retriever

# retriver= vector_store.as_retriever(search_type='similarity', search_kwargs={"k":4})
# print(retriver.invoke('what are neurons '))

# augmentation

 