#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.schema import OutputParserException
from langchain.schema.output_parser import StrOutputParser


# In[2]:


# Define the PromptTemplate
prompt = PromptTemplate(
    template="""
You are a travel assistant. Based on the following inputs, recommend a destination:
- Preferred activity: {activity}
- Budget (in USD): {budget}

Provide a response strictly in JSON format:
{{
    "destination": "<destination>",
    "activity": "<activity>",
    "cost": "<cost>"
}}
""",
    input_variables=["activity", "budget"],
)


# In[3]:


# Define the Output Parser
response_schemas = [
    ResponseSchema(name="destination", description="Recommended travel destination"),
    ResponseSchema(name="activity", description="Suggested activity at the destination"),
    ResponseSchema(name="cost", description="Estimated cost in USD for the trip"),
]
output_parser = StrOutputParser()


# In[4]:


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-0613", temperature=0)


# In[5]:


# Create the LangChain Expression (LLM Chain)
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)


# In[6]:


# Test the chain with an example
input_data = {"activity": "hiking", "budget": 1000}
result = chain.run(input_data)


# In[7]:


# Parse the structured output
parsed_result = output_parser.parse(result)


# In[8]:


# Display the result
print("Recommendation:", parsed_result)

