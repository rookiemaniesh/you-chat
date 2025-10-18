from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
##indexing
# doc loading
video_id='aircAruvnKk'
try:

    transcript_list= YouTubeTranscriptApi().fetch(video_id,languages=['en'])

    transcript_preprocesssed=transcript_list.to_raw_data()

    transcript=" ".join(chunk['text'] for chunk in transcript_preprocesssed)
    # print(transcript)
except TranscriptsDisabled:
    print("No Caption available for the video!!")

except Exception as e:
    print('an error occured {e}')

# text splitting

splitter=RecursiveCharacterTextSplitter(
    chunk_size=550,
    chunk_overlap=150
)
chunks=splitter.create_documents([transcript])
# print(chunks[2].page_content)
# print(len(chunks))


# embeddings

embeddings=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
vector_store=FAISS.from_documents(chunks,embeddings)
# print(vector_store.index_to_docstore_id)

# retriever

retriver= vector_store.as_retriever(search_type='similarity', search_kwargs={"k":4})


# augmentation
llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
prompt=PromptTemplate(
    template="""You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
      input_variables=['context','question']
    
    
)
question='what is neurons'
retrived_docs=retriver.invoke(question)
# context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
# final_prompt=prompt.invoke({'context':context_text,'question':question})
# print(final_prompt)

# generation
# answer = llm.invoke(final_prompt)
# print(answer.content)

# forming chain
def format_docs(retrived_docs):
    context_text="\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text

parallel_chain=RunnableParallel({
    'context':retriver|RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})
parser=StrOutputParser()
main_chain=parallel_chain | prompt | llm | parser
result=main_chain.invoke(question)
print(result)