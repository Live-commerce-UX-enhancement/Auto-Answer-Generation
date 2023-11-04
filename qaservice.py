from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from typing import Any, Dict, List
from langchain.docstore.document import Document

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os

import urllib.request, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class QAService:
    def __init__(self):

        embeddings = OpenAIEmbeddings()

        # MAKE VECTORSTORES
        self.vector_store = Chroma(embedding_function=embeddings)

        # MAKE QA Langchain
        system_template = """You are a host of a live shopping broadcast. You will be provided with ("SOURCES") and questions.
        When a Korean question sentence is entered as an input, please refer to the "SOURCES" and answer it.
        Please write your answer in a sentence. Please do not include greetings such as "안녕하세요", "안녕하세요 고객님" or “감사합니다” in your answer.
        If you don't know the answer, just say that "답변을 생성할 수 없습니다.", don't try to make up an answer.

        ----------------
        {summaries}

        You MUST answer in Korean with single sentence."""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        chain_type_kwargs = {"prompt": prompt}

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        self.chain = NoRetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

    # 상세 정보 저장
    def add_info(self, broadcast_id, information):
        # 방송 상세 정보 저장
        broadcast_info_list = information.broadcast
        broadcast_texts = list()
        broadcast_metadatas = list()
        for broadcast_info in broadcast_info_list:
            for broadcast_text in broadcast_info.texts:
                broadcast_texts.append(broadcast_info.type + '\n' + broadcast_text)
                broadcast_metadatas.append({'broadcast_id': broadcast_id, 'source': 'broadcast'})
        self.vector_store.add_texts(texts=broadcast_texts, metadatas=broadcast_metadatas)

        # 상품 상세 정보 저장
        product_info_list = information.product
        product_texts = list()
        product_metadatas = list()
        for product_info in product_info_list:
            for product_text in product_info.texts:
                product_texts.append(product_info.name + '\n' + product_text)
                product_metadatas.append({'broadcast_id': broadcast_id, 'source': 'product'})
        self.vector_store.add_texts(texts=product_texts, metadatas=product_metadatas)

        # 질의 응답 (QA) 데이터 셋 로드
        # self.add_qa_info(broadcast_id, product_info_list)

    def add_admin_answer_info(self, broadcast_id, question, answer):
        metadata = {'answer': answer, 'broadcast_id': broadcast_id, 'source': 'admin_qa'}

        self.vector_store.add_texts(texts=[question], metadatas=[metadata])

    def add_qa_info(self, broadcast_id, product_info_list):
        qa_texts = list()
        qa_metadatas = list()

        for product_info in product_info_list:
            product_id = product_info.id
            product_name = product_info.name
            qa_list_url = f'https://shopping.naver.com/shopv/v1/comments/PRODUCTINQUIRY/{product_id}'

            with urllib.request.urlopen(qa_list_url) as url:
                data = json.loads(url.read().decode())
                qa_size = data['totalElements']

            if qa_size > 0:
                qa_list_url_size = f'https://shopping.naver.com/shopv/v1/comments/PRODUCTINQUIRY/{product_id}?size={qa_size}&sellerAnswerYn=true'
                with urllib.request.urlopen(qa_list_url_size) as url:
                    data = json.loads(url.read().decode())

                    if data['totalElements'] > 0:
                        for content in data['contents']:
                            if not content['secretYn']:
                                qa_id = content['id']
                                qa_detail_url = f'https://shopping.naver.com/shopv/v1/comments/replies/{qa_id}'
                                with urllib.request.urlopen(qa_detail_url) as qa_url:
                                    qa_detail = json.loads(qa_url.read().decode())
                                    question = content['commentContent']
                                    answer = qa_detail[0]['commentContent']

                                    qa_texts.append(product_name + '\n' + question)
                                    qa_metadatas.append({'answer': answer, 'broadcast_id': broadcast_id, 'source': 'qa'})

        if len(qa_texts) > 0:
            self.vector_store.add_texts(texts=qa_texts, metadatas=qa_metadatas)


    # 문서 선택
    def select_documents(self, broadcast_id, query):

        res_docs = []

        result = self.vector_store.similarity_search_with_score(query, k=self.vector_store._collection.count(), filter={"broadcast_id":broadcast_id})[:3]

        for doc, score in result:
            if doc.metadata['source'] == 'qa' or doc.metadata['source'] == 'admin_qa':
                doc.page_content = "Question:" + doc.page_content + "\nAnswer:" + doc.metadata['answer']
                doc.metadata['score'] = score
            else:
                doc.metadata['score'] = score
            res_docs.append(doc)

        return res_docs

    # 답변 생성
    def get_answer(self, broadcast_id, query):
        result = dict()
        result['query'] = query
        docs = self.select_documents(broadcast_id, query)

        self.chain.update_docs(docs)
        answer = self.chain(query)
        result['answer'] = answer['answer']
        return result


class NoRetrievalQAWithSourcesChain(BaseQAWithSourcesChain):

    reduce_k_below_max_tokens: bool = False
    max_tokens_limit: int = 3375

    documents = list()

    def update_docs(self, docs):
      self.documents = docs

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_documents_chain.llm_chain.llm.get_num_tokens(
                    doc.page_content
                )
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun
    ) -> List[Document]:
        docs = self.documents
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun
    ) -> List[Document]:
        docs = await self.documents
        return self._reduce_tokens_below_limit(docs)