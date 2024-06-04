from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

from pydantic import BaseModel 
from typing import List 
from dotenv import load_dotenv


load_dotenv() # loads the .env file with OPENAI_API_KEY

OPENAI_MODEL = "gpt-3.5-turbo"
NUM_KEYWORDS = 250

class Plan(BaseModel):
    plan: List[List[str]]
    

class ATSKeywordGenerator:
    def __init__(self):
        print("Loading the data from the Directory 'data/'")
        loader = DirectoryLoader("data/", glob="*.txt", show_progress=True, loader_cls=TextLoader, 
                                 loader_kwargs={"autodetect_encoding": True})
        
        print("This step will take some time. Creating Vector Indexes from the loaded data.")
        self.index = VectorstoreIndexCreator(
                embedding=OpenAIEmbeddings(),
                vectorstore_cls=DocArrayInMemorySearch
            ).from_loaders([loader])
        
        self.llm = ChatOpenAI(temperature=0, model=OPENAI_MODEL)
        self.prepare_chain()

    def generate_keywords(self, role: str, num_keywords: int = NUM_KEYWORDS) -> str:
        '''
        Function for generating ATS keywords and grouping them with same meaning. 
        
        Args: 
            role: str - the keyword or role for which ATS keywords are to be generated.
            num_keywords: int = 500 - number of keywords to be considered (before refinement).
        Returns:
            str - returns the status of the generation. (SUCCESS or ERROR)
        '''
        
        try:
            role_description = self.index.query(f"All the requirements and description for the role {role}", 
                                                llm=self.llm)
            
            print("Fetching response from the LLM")
            response = self.overall_chain({
                "role": role, 
                "num_keywords": num_keywords, 
                "description": role_description
            })
            
            with open("result.json", "w") as file:
                file.write(self.parser.parse(
                    response['refined_keywords']
                ).model_dump_json())
                
            print("Results are stored at 'result.json' file.")
                
            return "SUCCESS"
            
        except Exception as e:
            print(e)
            return "ERROR"

    def prepare_chain(self):
        self.parser = PydanticOutputParser(pydantic_object=Plan)

        prompt_1 = PromptTemplate(
            template='''
            Create a list of {num_keywords} ATS
        keywords associated with the given role. The keywords should be as distinct 
        and unique as possible, and one element in the list of keyword should not be the extension 
        of another keyword. Do not have duplicates and do not repeat any keyword. 
        Use the given description for the role to get a better understanding 
        of the role.

        Role: {role}

        Description: {description}
            ''', 
            input_variables=['num_keywords', 'role', 'description']
        )

        prompt_2 = PromptTemplate(
            template=''' 
            From the given list of ATS keywords and role, cluster 
            the keywords with the exact same meaning in the same list. Use as many keywords as possible
            and each nested list should represent distinct meaning. 
            For the given list of keywords, create 'Plan' which is a list of 'Item' where 
            each Item represents a list of keywords with exact same meaning. 
            for example, given list [l1, l2, l3, l4] if l1 and l2 have same exact meaning and 
            l3 and l4 also have same exact meaning, the generated list will be Plan: [Item_1: [l1, l2], Item_2: [l3, l4]]. 
            Instead of eliminating the keyword, merge the keywords with same meaning 
            in the same list. If and only if a keyword have a distinct meaning than all the other keywords, it can be alone 
            in the list.
            
            Role: {role}
            
            Keywords: {keywords}
            
            Format Instructions: {format_instructions}
            ''', 
            input_variables=['role', 'keywords'], 
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        prompt_3 = PromptTemplate(
            template=''' 
            From the given nested list of ATS keywords, with the given role. Make sure that
            the ATS keywords in a same nested list must have exact same meaning. If the ATS keywords 
            in an Item does not have same meaning, create a new Item and append it to the Plan. 
            Use as many keywords as possible
            and each Item should represent distinct meaning.
            For example, given nested list - Plan: [Item_1: [l1, l2], Item_2: [l3, l4], Item_3: [l5]] 
            if l1 and l2 does not have same 
            meaning, the refined list will be - Plan: [Item_1: [l1], Item_2: [l3, l4], Item_3: [l5], Item_4: [l2]]
            Instead of eliminating the keyword, merge the keywords with same meaning 
            in the same list. If and only if a keyword have a distinct meaning than all the other keywords, it can be alone 
            in the list.
            
            The input and output format is same. 
            
            Role: {role}
            
            Keywords: {curated_keywords}
            
            Format Instructions: {format_instructions}
            ''',
            input_variables=['role', 'curated_keywords'],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain_1 = LLMChain(llm=self.llm, prompt=prompt_1, 
                        output_key="keywords")
        chain_2 = LLMChain(llm=self.llm, prompt=prompt_2, 
                        output_key="curated_keywords")
        chain_3 = LLMChain(llm=self.llm, prompt=prompt_3, 
                        output_key="refined_keywords")

        self.overall_chain = SequentialChain(
            chains=[chain_1, chain_2, chain_3], 
            input_variables=['role', 'num_keywords', 'description'], 
            output_variables=['keywords', 'curated_keywords', 'refined_keywords'],
            verbose=True
        )
        

if __name__ == "__main__":
    keyword_generator = ATSKeywordGenerator()
    role = input("What is the role?")
    # num_keywords = input("Number of keywords to be generated? (before refinement)")
    
    print(keyword_generator.generate_keywords(role))
    